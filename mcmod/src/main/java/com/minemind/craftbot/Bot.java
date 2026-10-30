package com.minemind.craftbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.tags.FluidTags;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.Container;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.crafting.CraftingInput;
import net.minecraft.world.item.crafting.CraftingRecipe;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraft.world.item.crafting.PlacementInfo;
import net.minecraft.world.item.crafting.RecipeHolder;
import net.minecraft.world.item.crafting.RecipeType;
import net.minecraft.world.item.crafting.ShapedRecipe;
import net.minecraft.world.item.crafting.SingleRecipeInput;
import net.minecraft.world.item.crafting.SmeltingRecipe;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.function.Predicate;

/**
 * One fake player driven by the policy. Action semantics mirror src/sim.cpp: each decision lasts 4 ticks,
 * except attack, which holds the button until the targeted block breaks, the target changes or 80 ticks pass.
 * Movement, mining, drops and placement are vanilla Minecraft. Crafting and smelting go through the server's
 * RecipeManager (vanilla recipes, vanilla ingredient and grid rules); the hand-coded recipes are only a fallback
 * for a server without crafting recipes. Chests are real chest block entities (27 slots, so deposit/withdraw/give
 * move what fits). Eating is never automatic: nothing in this class consumes food.
 */
public final class Bot {
	static final String[] ACTIONS = {"noop", "forward", "back", "left", "right", "jump", "forward_jump",
		"turn_left", "turn_right", "look_up", "look_down", "attack", "place_block", "place_table", "place_furnace",
		"craft_planks", "craft_sticks", "craft_table", "craft_wooden_pickaxe", "craft_stone_pickaxe",
		"craft_furnace", "smelt_iron", "craft_iron_pickaxe", "craft_chest", "place_chest", "deposit", "withdraw",
		"give"};
	static final int TICKS = 4, MAX_ATTACK_TICKS = 80, GIVE_RANGE = 4, DUMP_RADIUS = 16;
	/** Sim safety net: a bot below this height above the world bottom is in the void. */
	static final int VOID_MARGIN = 5;

	final String name;
	ServerPlayer player;
	final PolicyClient policy;

	int decisions;
	int fuel;
	float mineProgress;
	boolean paused = true;
	String task;            // sim item name the bot is working towards, or null = free play
	long taskStart;
	int lastAction;
	int msg;                // message token from the last decision (0 = silent), shown on the nametag
	// Training bots (Train.java): team is non-null, decisions come from the batched bridge reply.
	Train.Team team;
	int slot;
	double epRet;           // raw sim-style return of the current training episode
	boolean diedFlag;       // died since the last batch (reported in the batch info)
	byte[] trainImg;
	float[] trainVec;
	static boolean debug = Boolean.getBoolean("craftbot.debug");
	private final boolean[] got = new boolean[Obs.ITEM_NAMES.length];
	private final Vec3 spawnPos;
	private final float spawnYaw;

	private CompletableFuture<PolicyClient.Decision> pending;
	private CompletableFuture<Obs.Layout> probe;
	private boolean resetMemory = true;
	private int ticksLeft;
	private float fwd, strafe;
	private boolean jump, attacking;
	private int attackTicks;
	private BlockPos minePos;
	private long lastErrorTick = -1000;
	private byte[] img;
	private float[] vec;

	Bot(String name, ServerPlayer player, PolicyClient policy) {
		this.name = name;
		this.player = player;
		this.policy = policy;
		this.spawnPos = player.position();
		this.spawnYaw = player.getYRot();
		for (int i = 0; i < got.length; i++) got[i] = Obs.count(player.getInventory(), Obs.ITEMS[i]) > 0;
	}

	void say(MinecraftServer server, String msg) {
		server.getPlayerList().broadcastSystemMessage(Component.literal("[" + name + "] " + msg), false);
	}

	void tick(MinecraftServer server) {
		if (team == null && !paused && ticksLeft <= 0 && !attacking) decide(server);

		player.zza = fwd;
		player.xxa = -strafe;  // vanilla: positive xxa strafes left
		player.setJumping(jump);
		if (attacking) {
			attackTick(server);
			attackTicks++;
			if (attacking && attackTicks >= MAX_ATTACK_TICKS) stopAttack();
		} else {
			clearMining();
			if (ticksLeft > 0 && --ticksLeft == 0) {
				fwd = strafe = 0;
				jump = false;
			}
		}

		player.doTick();
		if (server.getTickCount() % 10 == 0) player.level().getChunkSource().move(player);
		if (team == null && server.getTickCount() % 5 == 0) checkProgress(server);
		if (team == null && needsRescue(player)) rescue(server);
	}

	private void decide(MinecraftServer server) {
		Obs.Layout L = policy.layout();
		if (L == null) {
			// First contact: ask the bridge for its checkpoint layout (res / VEC_DIM / blocks).
			if (probe == null) probe = policy.probe();
			if (probe.isDone()) {
				if (probe.isCompletedExceptionally()) reportError(server, "policy bridge unreachable - is mcbridge running?");
				probe = null;
			}
			ticksLeft = 20;
			return;
		}
		if (img == null || img.length != L.res * L.res * 2) {
			img = L.newImage();
			vec = L.newVec();
		}
		if (pending == null) {
			Level level = player.level();
			Obs.build(level, L, player.getX(), player.getY(), player.getZ(), player.getYRot(), player.getXRot(),
				player.onGround(), player.getHealth(), mineProgress, fuel, goalIndex(), decisions % Obs.MAX_STEPS,
				player.getInventory(), teammates(), bodies(level), img, vec);
			pending = policy.decide(name, L, img, vec, resetMemory);
			resetMemory = false;
		}
		if (!pending.isDone()) return;  // reply usually arrives next tick; idle meanwhile
		try {
			PolicyClient.Decision d = pending.join();
			setMsg(server, d.msg());
			start(d.action());
		} catch (Exception e) {
			resetMemory = true;
			reportError(server, "policy bridge error (" + e.getMessage() + ")");
			ticksLeft = 20;
		}
		pending = null;
	}

	private void reportError(MinecraftServer server, String text) {
		long now = server.getTickCount();
		if (now - lastErrorTick > 200) {
			say(server, text);
			lastErrorTick = now;
		}
	}

	/** Other craftbot bots in spawn order (slot = position). A bot in another level is an absent slot. */
	private List<Obs.Mate> teammates() {
		List<Obs.Mate> out = new ArrayList<>();
		for (Bot b : CraftBotMod.BOTS.values()) {
			if (b == this) continue;
			if (b.player.level() != player.level()) out.add(new Obs.Mate(1e9, 1e9, 1e9, 0));  // absent
			else out.add(new Obs.Mate(b.player.getX(), b.player.getY(), b.player.getZ(), b.msg));
		}
		return out;
	}

	/** Every other player in the level (bots included): rays see them as AGENT. */
	private List<Obs.Mate> bodies(Level level) {
		List<Obs.Mate> out = new ArrayList<>();
		for (Player p : level.players())
			if (p != player) out.add(new Obs.Mate(p.getX(), p.getY(), p.getZ(), 0));
		return out;
	}

	void force(int a) {
		paused = true;
		pending = null;
		start(a);
	}

	void setMsg(MinecraftServer server, int token) {
		int t = Math.max(0, Math.min(token, Obs.NUM_MSGS - 1));
		if (t == msg) return;
		msg = t;
		player.setCustomName(Component.literal(msg == 0 ? name : name + " m" + msg));
		player.setCustomNameVisible(true);
	}

	void start(int a) {
		lastAction = a;
		decisions++;
		if (decisions % Obs.MAX_STEPS == 0) resetMemory = true;  // sim episodes were 3000 decisions long
		fwd = strafe = 0;
		jump = false;
		ticksLeft = TICKS;
		Level level = player.level();
		switch (a) {
			case 1 -> fwd = 1;
			case 2 -> fwd = -1;
			case 3 -> strafe = -1;
			case 4 -> strafe = 1;
			case 5 -> jump = true;
			case 6 -> { fwd = 1; jump = true; }
			case 7 -> setYaw(player.getYRot() - 15f);
			case 8 -> setYaw(player.getYRot() + 15f);
			case 9 -> player.setXRot(Math.max(-90f, player.getXRot() - 15f));
			case 10 -> player.setXRot(Math.min(90f, player.getXRot() + 15f));
			case 11 -> startAttack();
			case 12 -> place(s -> s.is(ItemTags.STONE_TOOL_MATERIALS) || s.is(Items.DIRT) || s.is(ItemTags.PLANKS));
			case 13 -> place(s -> s.is(Items.CRAFTING_TABLE));
			case 14 -> place(s -> s.is(Items.FURNACE));
			case 15 -> craft(s -> s.is(ItemTags.PLANKS), this::craftPlanks);
			case 16 -> craft(s -> s.is(Items.STICK), () -> { if (take(Obs.ITEMS[1], 2)) give(Items.STICK, 4); });
			case 17 -> craft(s -> s.is(Items.CRAFTING_TABLE), () -> { if (take(Obs.ITEMS[1], 4)) give(Items.CRAFTING_TABLE, 1); });
			case 18 -> craft(s -> s.is(Items.WOODEN_PICKAXE), () -> craftPick(Obs.ITEMS[1], Items.WOODEN_PICKAXE));
			case 19 -> craft(s -> s.is(Items.STONE_PICKAXE), () -> craftPick(Obs.ITEMS[5], Items.STONE_PICKAXE));
			case 20 -> craft(s -> s.is(Items.FURNACE), () -> {
				if (Obs.near(level, player.getX(), player.getY(), player.getZ(), Blocks.CRAFTING_TABLE) && take(Obs.ITEMS[5], 8)) give(Items.FURNACE, 1);
			});
			case 21 -> smelt();
			case 22 -> craft(s -> s.is(Items.IRON_PICKAXE), () -> craftPick(Obs.ITEMS[10], Items.IRON_PICKAXE));
			case 23 -> craft(s -> s.is(Items.CHEST), () -> { if (nearTable() && take(Obs.ITEMS[1], 8)) give(Items.CHEST, 1); });
			case 24 -> place(s -> s.is(Items.CHEST));
			case 25 -> deposit();
			case 26 -> withdraw();
			case 27 -> give();
			default -> { }
		}
	}

	private boolean nearTable() {
		return Obs.near(player.level(), player.getX(), player.getY(), player.getZ(), Blocks.CRAFTING_TABLE);
	}

	private void setYaw(float yaw) {
		player.setYRot(yaw);
		player.setYHeadRot(yaw);
		player.yBodyRot = yaw;
	}

	private Obs.Hit aim() {
		return Obs.target(player.level(), player.getX(), player.getY(), player.getZ(), player.getYRot(), player.getXRot());
	}

	// ---------------------------------------------------------------- mining

	/** 11 attack: the button is held until the targeted block breaks, the target changes or the target is lost. */
	private void startAttack() {
		attacking = true;
		attackTicks = 0;
		minePos = null;
		mineProgress = 0;
	}

	private void stopAttack() {
		attacking = false;
		attackTicks = 0;
		ticksLeft = 0;
		clearMining();
	}

	private void attackTick(MinecraftServer server) {
		ServerLevel level = (ServerLevel) player.level();
		Obs.Hit h = aim();
		if (!h.hit) {
			if (debug) System.out.println("[craftbot] attack: nothing in reach");
			stopAttack();
			return;
		}
		BlockPos pos = new BlockPos(h.x, h.y, h.z);
		BlockState state = level.getBlockState(pos);
		if (minePos == null) {
			minePos = pos;
			equipBest(state);  // best tool for this block, from anywhere in the inventory
		} else if (!pos.equals(minePos)) {
			if (debug) System.out.println("[craftbot] attack: target changed, stop");
			stopAttack();
			return;
		}
		if (attackTicks % 4 == 0) player.swing(InteractionHand.MAIN_HAND, true);
		mineProgress += state.getDestroyProgress(player, level, pos);  // vanilla speed incl. tool + harvest level
		level.destroyBlockProgress(player.getId(), pos, Math.min(9, (int) (mineProgress * 10)));
		if (debug) System.out.printf("[craftbot] tick %d target %s %s prog %.3f onGround %s%n", attackTicks, pos, state, mineProgress, player.onGround());
		if (mineProgress >= 1f) {
			Inventory inv = player.getInventory();
			int[] before = team != null ? Train.counts(inv) : null;
			boolean ok = player.gameMode.destroyBlock(pos);
			if (debug) System.out.printf("[craftbot] destroyBlock -> %s, now %s%n", ok, level.getBlockState(pos));
			// The sim puts drops straight into the inventory; collect what just popped out.
			for (ItemEntity item : level.getEntitiesOfClass(ItemEntity.class, new AABB(pos).inflate(1.5))) {
				item.setNoPickUpDelay();
				item.playerTouch(player);
			}
			if (before != null) {  // training: items that came out of this block count as mined
				int[] after = Train.counts(inv);
				for (int i = 0; i < after.length; i++) if (after[i] > before[i]) team.gain(this, i);
			}
			stopAttack();  // the block broke: the held button is released
		}
	}

	private void clearMining() {
		if (minePos != null) ((ServerLevel) player.level()).destroyBlockProgress(player.getId(), minePos, -1);
		minePos = null;
		mineProgress = 0;
	}

	/**
	 * Moves the hotbar stack that mines this block fastest into the main hand. A tool that is the correct
	 * tool for the block (drops) beats any faster one that is not; the current selection wins ties.
	 */
	private void equipBest(BlockState state) {
		Inventory inv = player.getInventory();
		int sel = inv.getSelectedSlot();
		float bestScore = score(inv.getItem(sel), state);
		int best = -1;
		for (int i = 0; i < 36; i++) {
			ItemStack s = inv.getItem(i);
			if (s.isEmpty() || i == sel) continue;
			float sc = score(s, state);
			if (sc > bestScore) { bestScore = sc; best = i; }
		}
		if (best >= 0) moveToHand(best);
	}

	private static float score(ItemStack s, BlockState state) {
		if (s.isEmpty()) return 1f;  // bare hand
		boolean correct = !state.requiresCorrectToolForDrops() || s.isCorrectToolForDrops(state);
		return (correct ? 1000f : 0f) + s.getDestroySpeed(state);
	}

	/** Swaps inventory slot i with the selected hotbar slot (hotbar slots are only selected, not swapped). */
	private void moveToHand(int i) {
		Inventory inv = player.getInventory();
		int sel = inv.getSelectedSlot();
		if (i < 9) {
			inv.setSelectedSlot(i);
		} else {
			ItemStack cur = inv.getItem(sel);
			inv.setItem(sel, inv.getItem(i));
			inv.setItem(i, cur);
		}
	}

	// ---------------------------------------------------------------- placing

	private void place(Predicate<ItemStack> what) {
		Obs.Hit h = aim();
		if (!h.hit || h.face < 0) return;
		if (!hold(what)) return;
		Direction dir = Direction.getNearest(h.px - h.x, h.py - h.y, h.pz - h.z, null);
		if (dir == null) return;
		double[] d = Obs.lookDir(player.getYRot(), player.getXRot());
		Vec3 hit = new Vec3(player.getX() + d[0] * h.dist, player.getY() + Obs.EYE + d[1] * h.dist, player.getZ() + d[2] * h.dist);
		BlockHitResult bhr = new BlockHitResult(hit, dir, new BlockPos(h.x, h.y, h.z), false);
		// Sneaking makes useItemOn place the held block instead of opening the chest / table it points at.
		boolean wasSneaking = player.isShiftKeyDown();
		player.setShiftKeyDown(true);
		try {
			player.gameMode.useItemOn(player, player.level(), player.getMainHandItem(), InteractionHand.MAIN_HAND, bhr);
		} finally {
			player.setShiftKeyDown(wasSneaking);
		}
		player.swing(InteractionHand.MAIN_HAND, true);
	}

	/** Moves a matching stack into the main hand. */
	private boolean hold(Predicate<ItemStack> what) {
		Inventory inv = player.getInventory();
		int sel = inv.getSelectedSlot();
		if (what.test(inv.getItem(sel))) return true;
		for (int i = 0; i < 36; i++) {
			ItemStack s = inv.getItem(i);
			if (s.isEmpty() || !what.test(s)) continue;
			moveToHand(i);
			return true;
		}
		return false;
	}

	// ---------------------------------------------------------------- chests (23..27)

	/** 25 deposit: every non-tool stack of the agent goes into the chest under the crosshair (what fits). */
	private void deposit() {
		Obs.Hit h = aim();
		if (!h.hit || h.id != Obs.CHEST) return;
		Container chest = Obs.chestAt(player.level(), new BlockPos(h.x, h.y, h.z));
		if (chest == null) return;
		Container inv = player.getInventory();
		for (int i = 0; i < 36; i++) {
			ItemStack s = inv.getItem(i);
			if (s.isEmpty() || Obs.isTool(s)) continue;
			inv.setItem(i, insert(chest, s.copy()));
		}
	}

	/** 26 withdraw: every stack of the chest under the crosshair goes to the agent (what fits). */
	private void withdraw() {
		Obs.Hit h = aim();
		if (!h.hit || h.id != Obs.CHEST) return;
		Container chest = Obs.chestAt(player.level(), new BlockPos(h.x, h.y, h.z));
		if (chest == null) return;
		Inventory inv = player.getInventory();
		for (int j = 0; j < chest.getContainerSize(); j++) {
			ItemStack s = chest.getItem(j);
			if (s.isEmpty()) continue;
			ItemStack rest = s.copy();
			inv.add(rest);
			chest.setItem(j, rest.isEmpty() ? ItemStack.EMPTY : rest);
		}
		chest.setChanged();
	}

	/** 27 give: nearest other bot or player within 4 blocks gets half (rounded up) of each non-tool stack. */
	private void give() {
		Level level = player.level();
		Player target = null;
		double best = GIVE_RANGE * GIVE_RANGE + 1e-9;
		for (Player p : level.players()) {
			if (p == player) continue;
			double dx = p.getX() - player.getX(), dy = p.getY() - player.getY(), dz = p.getZ() - player.getZ();
			double d2 = dx * dx + dy * dy + dz * dz;
			if (d2 <= best) { best = d2; target = p; }
		}
		if (target == null) return;
		Inventory inv = player.getInventory();
		Inventory tinv = target.getInventory();
		for (int i = 0; i < 36; i++) {
			ItemStack s = inv.getItem(i);
			if (s.isEmpty() || Obs.isTool(s)) continue;
			int n = (s.getCount() + 1) / 2;
			ItemStack part = s.copyWithCount(n);
			tinv.add(part);
			s.shrink(n - part.getCount());   // part keeps what the receiver could not fit
			inv.setItem(i, s);
		}
	}

	/** Puts a stack into a container (merging, then empty slots); returns what did not fit. */
	private static ItemStack insert(Container c, ItemStack s) {
		ItemStack rem = s;
		for (int j = 0; j < c.getContainerSize() && !rem.isEmpty(); j++) {
			ItemStack cur = c.getItem(j);
			if (!cur.isEmpty() && ItemStack.isSameItemSameComponents(cur, rem)) {
				int mv = Math.min(cur.getMaxStackSize() - cur.getCount(), rem.getCount());
				if (mv > 0) { cur.grow(mv); rem.shrink(mv); }
			}
		}
		for (int j = 0; j < c.getContainerSize() && !rem.isEmpty(); j++) {
			if (c.getItem(j).isEmpty()) {
				int mv = Math.min(rem.getMaxStackSize(), rem.getCount());
				c.setItem(j, rem.copyWithCount(mv));
				rem.shrink(mv);
			}
		}
		c.setChanged();
		return rem;
	}

	// ---------------------------------------------------------------- crafting

	/**
	 * Crafts through the server RecipeManager when it has crafting recipes; the hand-coded version runs only
	 * when the server has none (so a recipe that is merely not affordable or not in reach is simply a no-op).
	 */
	private void craft(Predicate<ItemStack> wanted, Runnable handCoded) {
		if (craftVanilla(wanted) || hasVanillaCrafting()) return;
		handCoded.run();
	}

	private boolean hasVanillaCrafting() {
		if (!(player.level() instanceof ServerLevel level)) return false;
		for (RecipeHolder<?> h : level.getServer().getRecipeManager().getRecipes())
			if (h.value() instanceof CraftingRecipe) return true;
		return false;
	}

	/**
	 * One craft of the first vanilla crafting recipe whose result satisfies {@code wanted} and whose ingredients
	 * are in the inventory. Recipes bigger than 2x2 need a crafting table in reach, as in vanilla (the inventory
	 * grid is 2x2). Returns true if an item was crafted.
	 */
	private boolean craftVanilla(Predicate<ItemStack> wanted) {
		if (!(player.level() instanceof ServerLevel level)) return false;
		Inventory inv = player.getInventory();
		int[] taken = new int[36];
		for (RecipeHolder<?> holder : level.getServer().getRecipeManager().getRecipes()) {
			if (!(holder.value() instanceof CraftingRecipe recipe)) continue;
			PlacementInfo placement = recipe.placementInfo();
			if (placement.isImpossibleToPlace()) continue;
			List<Optional<Ingredient>> grid;
			int w, h;
			if (recipe instanceof ShapedRecipe shaped) {
				w = shaped.getWidth();
				h = shaped.getHeight();
				grid = shaped.getIngredients();
			} else {
				// Shapeless: ingredients go in the first n cells of the smallest square grid that holds them.
				List<Ingredient> ings = placement.ingredients();
				if (ings.size() > 9) continue;
				w = h = ings.size() <= 4 ? 2 : 3;
				grid = new ArrayList<>();
				for (Ingredient ing : ings) grid.add(Optional.of(ing));
				while (grid.size() < w * h) grid.add(Optional.empty());
			}
			if ((w > 2 || h > 2) && !nearTable()) continue;

			// Pick one matching stack per ingredient (never more than a slot holds).
			Arrays.fill(taken, 0);
			int[] from = new int[grid.size()];
			boolean ok = true;
			for (int g = 0; g < grid.size() && ok; g++) {
				from[g] = -1;
				Optional<Ingredient> ing = grid.get(g);
				if (ing.isEmpty()) continue;
				for (int j = 0; j < 36; j++) {
					ItemStack s = inv.getItem(j);
					if (s.isEmpty() || s.getCount() - taken[j] <= 0 || !ing.get().test(s)) continue;
					from[g] = j;
					taken[j]++;
					break;
				}
				ok = from[g] >= 0;
			}
			if (!ok) continue;

			List<ItemStack> items = new ArrayList<>(grid.size());
			for (int g = 0; g < grid.size(); g++)
				items.add(from[g] < 0 ? ItemStack.EMPTY : inv.getItem(from[g]).copyWithCount(1));
			CraftingInput input = CraftingInput.of(w, h, items);
			if (!recipe.matches(input, level)) continue;
			ItemStack out = recipe.assemble(input);
			if (out.isEmpty() || !wanted.test(out)) continue;

			for (int g = 0; g < grid.size(); g++) if (from[g] >= 0) inv.getItem(from[g]).shrink(1);
			for (ItemStack rem : recipe.getRemainingItems(input)) if (!rem.isEmpty()) inv.add(rem);
			grant(out);
			return true;
		}
		return false;
	}

	/** Puts a crafted or smelted result into the inventory (dropping what does not fit); training counts it. */
	private void grant(ItemStack stack) {
		if (stack.isEmpty()) return;
		if (team != null) {  // training: a crafted or smelted item counts as obtained
			int idx = Train.itemIndex(stack.getItem());
			if (idx >= 0) team.gain(this, idx);
		}
		if (!player.getInventory().add(stack) && !stack.isEmpty()) player.drop(stack, false);
	}

	private void give(Item item, int n) {
		grant(new ItemStack(item, n));
	}

	private boolean take(Predicate<ItemStack> what, int n) {
		if (Obs.count(player.getInventory(), what) < n) return false;
		var inv = player.getInventory();
		for (int i = 0; i < inv.getContainerSize() && n > 0; i++) {
			ItemStack s = inv.getItem(i);
			if (s.isEmpty() || !what.test(s)) continue;
			int k = Math.min(n, s.getCount());
			s.shrink(k);
			n -= k;
		}
		return true;
	}

	// Hand-coded fallbacks (only used when the server has no crafting recipes at all).

	private void craftPlanks() {
		var inv = player.getInventory();
		for (int i = 0; i < inv.getContainerSize(); i++) {
			ItemStack s = inv.getItem(i);
			if (s.isEmpty() || !s.is(ItemTags.LOGS)) continue;
			Item planks = planksFor(s.getItem());
			s.shrink(1);
			give(planks, 4);
			return;
		}
	}

	private static Item planksFor(Item log) {
		var key = BuiltInRegistries.ITEM.getKey(log);
		String path = key.getPath().replace("stripped_", "")
			.replace("_log", "_planks").replace("_wood", "_planks").replace("_stem", "_planks").replace("_hyphae", "_planks");
		return BuiltInRegistries.ITEM.getOptional(key.withPath(path)).orElse(Items.OAK_PLANKS);
	}

	private void craftPick(Predicate<ItemStack> head, Item out) {
		if (!nearTable()) return;
		if (Obs.count(player.getInventory(), head) < 3 || Obs.count(player.getInventory(), Obs.ITEMS[2]) < 2) return;
		take(head, 3);
		take(Obs.ITEMS[2], 2);
		give(out, 1);
	}

	/** 21 smelt_iron: a furnace in reach, coal (8 items per coal) or a plank as fuel, the output from the SMELTING recipe. */
	private void smelt() {
		Level level = player.level();
		if (!Obs.near(level, player.getX(), player.getY(), player.getZ(), Blocks.FURNACE)
			|| Obs.count(player.getInventory(), Obs.ITEMS[9]) < 1) return;
		if (fuel == 0 && take(Obs.ITEMS[8], 1)) fuel = 8;          // coal smelts 8
		else if (fuel == 0 && take(Obs.ITEMS[1], 1)) fuel = 1;     // a plank smelts 1 (vanilla 1.5)
		if (fuel > 0) {
			fuel--;
			if (take(Obs.ITEMS[9], 1)) grant(smeltOutput(level, Items.RAW_IRON));
		}
	}

	private static ItemStack smeltOutput(Level level, Item raw) {
		if (level instanceof ServerLevel sl) {
			SingleRecipeInput input = new SingleRecipeInput(new ItemStack(raw));
			Optional<RecipeHolder<SmeltingRecipe>> rec = sl.getServer().getRecipeManager()
				.getRecipeFor(RecipeType.SMELTING, input, level);
			if (rec.isPresent()) {
				ItemStack out = rec.get().value().assemble(input);
				if (!out.isEmpty()) return out;
			}
		}
		return new ItemStack(Items.IRON_INGOT);
	}

	// ---------------------------------------------------------------- safety net (training / void / lava)

	/**
	 * True if the player is in lava, below the world's build floor + {@value #VOID_MARGIN} blocks (the void), or
	 * suffocating inside a block. Train.java calls this and resets such bots to their team spot.
	 */
	static boolean needsRescue(ServerPlayer p) {
		if (p.isDeadOrDying() || p.isRemoved()) return false;
		Level level = p.level();
		if (p.getY() < level.getMinY() + VOID_MARGIN) return true;
		if (p.isInLava()) return true;
		BlockPos feet = p.blockPosition();
		if (level.getFluidState(feet).is(FluidTags.LAVA) || level.getFluidState(feet.above()).is(FluidTags.LAVA)) return true;
		return p.isInWall();
	}

	/** Play-mode bots return to the spot they were spawned at. Training bots are reset by Train.java. */
	private void rescue(MinecraftServer server) {
		if (debug) System.out.println("[craftbot] " + name + " rescue from " + player.position());
		resetState();
		player.snapTo(spawnPos.x, spawnPos.y, spawnPos.z, spawnYaw, 0f);
		player.setDeltaMovement(Vec3.ZERO);
		say(server, "fell into lava or the void - back to spawn");
	}

	// ---------------------------------------------------------------- dump (FT-3, format minemind-parity-1)

	/** Bots of the same team other than this one (play mode: the other play bots), in slot / spawn order. */
	private List<Bot> sameTeamOthers() {
		List<Bot> out = new ArrayList<>();
		for (Bot b : CraftBotMod.BOTS.values()) if (b != this && b.team == team) out.add(b);
		out.sort(Comparator.comparingInt(b -> b.team == null ? 0 : b.slot));
		return out;
	}

	/** Index of a bot in the dump: its team slot, or its spawn-order position in play mode. */
	private static int dumpIndex(Bot b) {
		if (b.team != null) return b.slot;
		int i = 0;
		for (Bot o : CraftBotMod.BOTS.values()) {
			if (o == b) return i;
			if (o.team == null) i++;
		}
		return 0;
	}

	private static void addCount(JsonObject o, String key, int n) {
		o.addProperty(key, (o.has(key) ? o.get(key).getAsInt() : 0) + n);
	}

	/**
	 * Writes the FT-3 parity dump ("minemind-parity-1", spec in tools/parity/DUMP_FORMAT.md): pose, player, goal,
	 * inventory, teammates, bodies, chests, the radius-16 block cube and the obs computed from this same state.
	 * Teammates are the bots of this bot's team; bodies lists every other player in the level with kind
	 * "team" / "bot" / "player" (the obs rays hit all of them as AGENT, as Obs.build does).
	 */
	Path dump(MinecraftServer server, Path dir) throws IOException {
		ServerLevel level = (ServerLevel) player.level();
		Obs.Layout L = policy.layout() != null ? policy.layout() : Obs.Layout.V6;
		Inventory inv = player.getInventory();
		double px = player.getX(), py = player.getY(), pz = player.getZ();
		int goal = goalIndex();
		int step = decisions % Obs.MAX_STEPS;

		List<Bot> mates = sameTeamOthers();
		List<Obs.Mate> mateObs = new ArrayList<>();
		for (Bot b : mates) {
			if (b.player.level() != level) mateObs.add(new Obs.Mate(1e9, 1e9, 1e9, 0));  // absent slot
			else mateObs.add(new Obs.Mate(b.player.getX(), b.player.getY(), b.player.getZ(), b.msg));
		}
		List<Obs.Mate> bodyObs = bodies(level);
		byte[] oImg = L.newImage();
		float[] oVec = L.newVec();
		Obs.build(level, L, px, py, pz, player.getYRot(), player.getXRot(), player.onGround(), player.getHealth(),
			mineProgress, fuel, goal, step, inv, mateObs, bodyObs, oImg, oVec);

		JsonObject root = new JsonObject();
		root.addProperty("format", "minemind-parity-1");
		root.addProperty("source", "minecraft");
		root.addProperty("created_utc", Instant.now().truncatedTo(ChronoUnit.SECONDS).toString());

		JsonObject game = new JsonObject();
		game.addProperty("mc_version", server.getServerVersion());
		game.addProperty("mod", "craftbot " + FabricLoader.getInstance().getModContainer("craftbot")
			.map(m -> m.getMetadata().getVersion().getFriendlyString()).orElse("?"));
		game.addProperty("level", level.dimension().identifier().toString());
		game.addProperty("game_tick", server.getTickCount());
		game.add("seed", JsonNull.INSTANCE);
		root.add("game", game);

		JsonObject bot = new JsonObject();
		bot.addProperty("name", name);
		bot.addProperty("team", team == null ? null : team.index);
		bot.addProperty("index", dumpIndex(this));
		root.add("bot", bot);

		JsonObject pose = new JsonObject();
		pose.addProperty("x", px);
		pose.addProperty("y", py);
		pose.addProperty("z", pz);
		pose.addProperty("yaw", (double) player.getYRot());
		pose.addProperty("pitch", (double) player.getXRot());
		pose.addProperty("on_ground", player.onGround());
		pose.addProperty("eye_height", Obs.EYE);
		pose.addProperty("sneaking", player.isShiftKeyDown());
		root.add("pose", pose);

		JsonObject p = new JsonObject();
		p.addProperty("health", player.getHealth());
		p.addProperty("fuel", fuel);
		p.addProperty("mining_progress", mineProgress);
		if (attacking && minePos != null) {
			JsonArray t = new JsonArray();
			t.add(minePos.getX()); t.add(minePos.getY()); t.add(minePos.getZ());
			p.add("mining_target", t);
		} else {
			p.add("mining_target", JsonNull.INSTANCE);
		}
		p.addProperty("step", step);
		p.addProperty("max_steps", Obs.MAX_STEPS);
		p.addProperty("hunger", player.getFoodData().getFoodLevel());
		p.addProperty("saturation", (double) player.getFoodData().getSaturationLevel());
		root.add("player", p);

		JsonObject goalObj = new JsonObject();
		goalObj.addProperty("name", Obs.ITEM_NAMES[goal]);
		goalObj.addProperty("index", goal);
		root.add("goal", goalObj);

		// Mapped MineMind items (first matching ITEMS predicate); everything else by vanilla id.
		JsonObject mapped = new JsonObject(), other = new JsonObject();
		for (int j = 0; j < inv.getContainerSize(); j++) {
			ItemStack s = inv.getItem(j);
			if (s.isEmpty()) continue;
			int k = -1;
			for (int i = 0; i < Obs.ITEMS.length && k < 0; i++) if (Obs.ITEMS[i].test(s)) k = i;
			if (k >= 0) addCount(mapped, Obs.ITEM_NAMES[k], s.getCount());
			else addCount(other, BuiltInRegistries.ITEM.getKey(s.getItem()).toString(), s.getCount());
		}
		root.add("inventory", mapped);
		root.add("inventory_other", other);

		JsonArray teammates = new JsonArray();
		for (Bot b : mates) {
			if (b.player.level() != level) continue;  // absent slot: not written
			JsonObject t = new JsonObject();
			t.addProperty("index", dumpIndex(b));
			t.addProperty("x", b.player.getX());
			t.addProperty("y", b.player.getY());
			t.addProperty("z", b.player.getZ());
			t.addProperty("yaw", (double) b.player.getYRot());
			t.addProperty("pitch", (double) b.player.getXRot());
			t.addProperty("msg", b.msg);
			t.addProperty("health", b.player.getHealth());
			teammates.add(t);
		}
		root.add("teammates", teammates);

		JsonArray bodies = new JsonArray();
		for (Player other2 : level.players()) {
			if (other2 == player) continue;
			Bot ob = null;
			for (Bot b : CraftBotMod.BOTS.values()) if (b.player == other2) ob = b;
			JsonObject o = new JsonObject();
			o.addProperty("x", other2.getX());
			o.addProperty("y", other2.getY());
			o.addProperty("z", other2.getZ());
			o.addProperty("kind", ob == null ? "player" : (ob.team == team ? "team" : "bot"));
			bodies.add(o);
		}
		root.add("bodies", bodies);

		// Blocks: radius-16 cube around the feet block, clipped to the build limits in y.
		int R = DUMP_RADIUS;
		int fx = (int) Math.floor(px), fy = (int) Math.floor(py), fz = (int) Math.floor(pz);
		int minY = level.getMinY(), maxY = minY + level.getHeight() - 1;
		int x0 = fx - R, z0 = fz - R, y0 = Math.max(minY, fy - R), y1 = Math.min(maxY, fy + R);
		int nx = 2 * R + 1, nz = nx, ny = y1 - y0 + 1;
		int n = nx * nz * ny;
		List<String> palette = new ArrayList<>();
		Map<String, Integer> palIdx = new HashMap<>();
		ByteBuffer van = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN);
		byte[] ours = new byte[n];
		List<BlockPos> chestPos = new ArrayList<>();
		BlockPos.MutableBlockPos mp = new BlockPos.MutableBlockPos();
		int idx = 0;
		for (int y = y0; y <= y1; y++)
			for (int z = z0; z < z0 + nz; z++)
				for (int x = x0; x < x0 + nx; x++, idx++) {
					mp.set(x, y, z);
					String id;
					if (!level.hasChunkAt(mp)) {
						id = "unloaded";
						ours[idx] = (byte) 255;
					} else {
						BlockState st = level.getBlockState(mp);
						id = BuiltInRegistries.BLOCK.getKey(st.getBlock()).toString();
						int m = Obs.blockId(st);
						ours[idx] = (byte) m;
						if (m == Obs.CHEST) chestPos.add(mp.immutable());
					}
					Integer pi = palIdx.get(id);
					if (pi == null) {
						pi = palette.size();
						palette.add(id);
						palIdx.put(id, pi);
					}
					van.putShort((short) (int) pi);
				}
		JsonObject blocks = new JsonObject();
		blocks.addProperty("radius", R);
		JsonArray origin = new JsonArray();
		origin.add(x0); origin.add(y0); origin.add(z0);
		blocks.add("origin", origin);
		JsonArray shape = new JsonArray();
		shape.add(ny); shape.add(nz); shape.add(nx);
		blocks.add("shape", shape);
		JsonObject limits = new JsonObject();
		limits.addProperty("min_y", minY);
		limits.addProperty("max_y", maxY);
		blocks.add("build_limit", limits);
		JsonArray pal = new JsonArray();
		for (String s : palette) pal.add(s);
		blocks.add("palette", pal);
		blocks.addProperty("vanilla", Base64.getEncoder().encodeToString(van.array()));
		blocks.addProperty("ours", Base64.getEncoder().encodeToString(ours));
		root.add("blocks", blocks);

		JsonArray chests = new JsonArray();
		for (BlockPos cp : chestPos) {
			JsonObject c = new JsonObject();
			c.addProperty("x", cp.getX());
			c.addProperty("y", cp.getY());
			c.addProperty("z", cp.getZ());
			JsonObject items = new JsonObject();
			Container cont = Obs.chestAt(level, cp);
			if (cont != null)
				for (int i = 0; i < Obs.ITEM_NAMES.length; i++) {
					int k = Obs.count(cont, Obs.ITEMS[i]);
					if (k > 0) items.addProperty(Obs.ITEM_NAMES[i], k);
				}
			c.add("items", items);
			chests.add(c);
		}
		root.add("chests", chests);

		JsonObject vision = new JsonObject();
		vision.addProperty("res", L.res);
		vision.addProperty("fov_deg", Obs.FOV_DEG);
		vision.addProperty("view_dist", Obs.VIEW_DIST);
		vision.addProperty("reach", Obs.REACH);
		vision.addProperty("ray_origin", "feet + (0, 1.62, 0)");
		vision.addProperty("bodies_in_rays", true);
		vision.addProperty("depth_encoding", "min(255, floor(dist / view_dist * 255)); sky = 255");
		root.add("vision", vision);

		JsonObject obs = new JsonObject();
		JsonObject img = new JsonObject();
		JsonArray imgShape = new JsonArray();
		imgShape.add(L.res); imgShape.add(L.res); imgShape.add(2);
		img.add("shape", imgShape);
		img.addProperty("order", "row-major, row 0 = top; channels = (block_id, depth)");
		img.addProperty("encoding", "base64-u8");
		img.addProperty("data", Base64.getEncoder().encodeToString(oImg));
		obs.add("img", img);
		JsonObject vec = new JsonObject();
		vec.addProperty("dim", oVec.length);
		JsonArray names = new JsonArray();
		for (String s : vecNames()) names.add(s);
		vec.add("names", names);
		JsonArray values = new JsonArray();
		for (float f : oVec) values.add(f);
		vec.add("values", values);
		obs.add("vec", vec);
		obs.addProperty("source", "mcmod Obs.build (" + (L.v6 ? "v6" : "v5") + ", res " + L.res + ")");
		root.add("obs", obs);

		Files.createDirectories(dir);
		String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
		Path file = dir.resolve(stamp + "-" + name + ".json");
		Files.writeString(file, root.toString(), StandardCharsets.UTF_8);
		return file;
	}

	/** Names of the v6 vector obs, same order as craftsim.VEC_NAMES (102 entries). */
	static List<String> vecNames() {
		List<String> out = new ArrayList<>();
		for (String it : Obs.ITEM_NAMES) out.add("inv_" + it);
		out.addAll(List.of("health", "on_ground", "pitch", "sin_yaw", "cos_yaw", "mining_progress", "height",
			"near_table", "near_furnace", "fuel", "time_frac"));
		for (String it : Obs.ITEM_NAMES) out.add("goal_" + it);
		for (int j = 0; j < 4; j++) {
			out.addAll(List.of("tm" + j + "_present", "tm" + j + "_dx", "tm" + j + "_dy", "tm" + j + "_dz"));
			for (int m = 0; m < Obs.NUM_MSGS; m++) out.add("tm" + j + "_msg" + m);
		}
		out.addAll(List.of("chest_present", "chest_dx", "chest_dy", "chest_dz"));
		for (int i = 0; i < 9; i++) out.add("xhair_chest_" + Obs.ITEM_NAMES[i]);
		return out;
	}

	/** Used by CraftBotMod for /aibot dump. */
	static Path dumpDir() {
		return FabricLoader.getInstance().getGameDir().resolve("minemind_dumps");
	}

	// ---------------------------------------------------------------- progress reporting

	private void checkProgress(MinecraftServer server) {
		double secs = (server.getTickCount() - taskStart) / 20.0;
		for (int i = 0; i < got.length; i++) {
			if (got[i] || Obs.count(player.getInventory(), Obs.ITEMS[i]) == 0) continue;
			got[i] = true;
			say(server, String.format("got %s  (%.1fs, %d decisions)", Obs.ITEM_NAMES[i], secs, decisions));
			if (Obs.ITEM_NAMES[i].equals(task)) {
				say(server, String.format("TASK COMPLETE: %s in %.1fs - stopping. Give a new /aibot task.", task, secs));
				paused = true;
				fwd = strafe = 0;
				jump = false;
				stopAttack();
			}
		}
	}

	/** Sim ITEMS index of the current goal (the one-hot in the obs); log when there is no task. */
	int goalIndex() {
		if (team != null) return team.goal;
		if (task == null) return 0;
		int idx = Arrays.asList(Obs.ITEM_NAMES).indexOf(task);
		return idx < 0 ? 0 : idx;
	}

	void startTask(MinecraftServer server, String item) {
		task = item;
		taskStart = server.getTickCount();
		// The goal changed: drop the GRU memory and any in-flight decision made for the old goal.
		resetMemory = true;
		pending = null;
		for (int i = 0; i < got.length; i++) got[i] = Obs.count(player.getInventory(), Obs.ITEMS[i]) > 0;
		int idx = Arrays.asList(Obs.ITEM_NAMES).indexOf(item);
		if (idx >= 0 && got[idx]) {
			say(server, "already has " + item);
			return;
		}
		paused = false;
		say(server, "task: get " + item);
	}

	// ---------------------------------------------------------------- training helpers (Train.java)

	/** Training: the policy's next decision may be applied (the last action's window has ended). */
	boolean ready() {
		return ticksLeft <= 0 && !attacking;
	}

	/** Training: drop any running action (no walking, no attacking, no mining progress). */
	void resetState() {
		fwd = strafe = 0;
		jump = false;
		ticksLeft = 0;
		if (attacking) stopAttack();
		clearMining();
	}

	/** Training: a respawned fake player takes over this bot slot. */
	void replacePlayer(ServerPlayer p) {
		player = p;
		resetState();
		mineProgress = 0;
	}

	String status() {
		StringBuilder inv = new StringBuilder();
		for (int i = 0; i < got.length; i++) {
			int n = Obs.count(player.getInventory(), Obs.ITEMS[i]);
			if (n > 0) inv.append(Obs.ITEM_NAMES[i]).append('=').append(n).append(' ');
		}
		return String.format("%s %s task=%s pos=(%.1f, %.1f, %.1f) hp=%.0f decisions=%d last=%s msg=%d inv: %s",
			name, paused ? "PAUSED" : "running", task, player.getX(), player.getY(), player.getZ(), player.getHealth(),
			decisions, ACTIONS[lastAction], msg, inv.length() > 0 ? inv : "empty");
	}
}
