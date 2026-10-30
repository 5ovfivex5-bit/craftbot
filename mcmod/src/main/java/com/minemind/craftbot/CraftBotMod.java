package com.minemind.craftbot;

import com.mojang.authlib.GameProfile;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.commands.SharedSuggestionProvider;
import net.minecraft.core.UUIDUtil;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.world.level.GameType;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * /aibot spawn [name]      spawn a policy-driven fake player 2 blocks in front of you (facing you)
 * /aibot task <item> [name]  give a goal to every bot, or one named bot (log, planks, crafting_table, ...)
 * /aibot stop | go         pause / resume all bots
 * /aibot status            print bots, positions and inventories
 * /aibot dump <bot>        parity dump (pose, inventory, radius-16 blocks, obs) to <game dir>/minemind_dumps/
 * /aibot remove [name]     remove one bot (or all)
 * /aibot train start [teams] [agents] | stop | status | speed <tps>|sprint | goals <item,...>|default
 *                          online PPO fine-tuning in the real world (see Train.java, PROTOCOL 11)
 */
public final class CraftBotMod implements ModInitializer {
	static final Map<String, Bot> BOTS = new LinkedHashMap<>();
	static final PolicyClient POLICY = new PolicyClient();

	@Override
	public void onInitialize() {
		CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> dispatcher.register(
			Commands.literal("aibot")
				.then(Commands.literal("spawn")
					.executes(c -> spawn(c, "AIBot"))
					.then(Commands.argument("name", StringArgumentType.word()).executes(c -> spawn(c, StringArgumentType.getString(c, "name")))))
				.then(Commands.literal("task")
					.then(Commands.argument("item", StringArgumentType.word())
						.suggests((c, b) -> SharedSuggestionProvider.suggest(Obs.ITEM_NAMES, b))
						.executes(c -> task(c, null))
						.then(Commands.argument("name", StringArgumentType.word())
							.suggests((c, b) -> SharedSuggestionProvider.suggest(BOTS.keySet().toArray(new String[0]), b))
							.executes(c -> task(c, StringArgumentType.getString(c, "name"))))))
				.then(Commands.literal("stop").executes(c -> setPaused(c, true)))
				.then(Commands.literal("go").executes(c -> setPaused(c, false)))
				.then(Commands.literal("status").executes(CraftBotMod::status))
				.then(Commands.literal("act")
					.then(Commands.argument("action", StringArgumentType.word())
						.suggests((c, b) -> SharedSuggestionProvider.suggest(Bot.ACTIONS, b))
						.executes(CraftBotMod::act)))
				.then(Commands.literal("look")
					.then(Commands.argument("pitch", com.mojang.brigadier.arguments.FloatArgumentType.floatArg(-90, 90))
						.executes(c -> { float p = com.mojang.brigadier.arguments.FloatArgumentType.getFloat(c, "pitch");
							for (Bot b : BOTS.values()) b.player.setXRot(p); return 1; })))
				.then(Commands.literal("train")
					.then(Commands.literal("start")
						.executes(c -> trainCmd(c, () -> Train.start(c.getSource(), Train.DEFAULT_TEAMS, Train.DEFAULT_AGENTS)))
						.then(Commands.argument("teams", com.mojang.brigadier.arguments.IntegerArgumentType.integer(1, Train.MAX_TEAMS))
							.executes(c -> trainCmd(c, () -> Train.start(c.getSource(),
								com.mojang.brigadier.arguments.IntegerArgumentType.getInteger(c, "teams"), Train.DEFAULT_AGENTS)))
							.then(Commands.argument("agents", com.mojang.brigadier.arguments.IntegerArgumentType.integer(1, Train.MAX_AGENTS))
								.executes(c -> trainCmd(c, () -> Train.start(c.getSource(),
									com.mojang.brigadier.arguments.IntegerArgumentType.getInteger(c, "teams"), com.mojang.brigadier.arguments.IntegerArgumentType.getInteger(c, "agents")))))))
					.then(Commands.literal("stop")
						.executes(c -> trainCmd(c, () -> {
							Train.stop(c.getSource().getServer(), "stopped by /aibot train stop");
							c.getSource().sendSuccess(() -> Component.literal("training stopped; the bridge saves a checkpoint"), true);
						})))
					.then(Commands.literal("status").executes(c -> trainCmd(c, () -> Train.status(c.getSource()))))
					.then(Commands.literal("speed")
						.then(Commands.literal("sprint")
							.executes(c -> trainCmd(c, () -> Train.sprint(c.getSource(), c.getSource().getServer()))))
						.then(Commands.argument("tps", com.mojang.brigadier.arguments.FloatArgumentType.floatArg(1f, 500f))
							.executes(c -> trainCmd(c, () -> Train.speed(c.getSource(), c.getSource().getServer(),
								com.mojang.brigadier.arguments.FloatArgumentType.getFloat(c, "tps"))))))
					.then(Commands.literal("goals")
						.then(Commands.argument("items", StringArgumentType.greedyString())
							.executes(c -> trainCmd(c, () -> Train.goals(c.getSource(), StringArgumentType.getString(c, "items")))))))
				.then(Commands.literal("dump")
					.then(Commands.argument("name", StringArgumentType.word())
						.suggests((c, b) -> SharedSuggestionProvider.suggest(BOTS.keySet().toArray(new String[0]), b))
						.executes(CraftBotMod::dump)))
				.then(Commands.literal("remove")
					.executes(c -> remove(c, null))
					.then(Commands.argument("name", StringArgumentType.word()).executes(c -> remove(c, StringArgumentType.getString(c, "name")))))));

		ServerTickEvents.END_SERVER_TICK.register(server -> {
			try {
				Train.tick(server);
			} catch (Throwable t) {
				Train.onTickError(server, t);
			}
			for (Bot bot : new ArrayList<>(BOTS.values())) {
				if (bot.player.isRemoved() || bot.player.isDeadOrDying()) {
					bot.say(server, bot.player.isDeadOrDying() ? "died - removing" : "removed");
					drop(server, bot);
					continue;
				}
				bot.tick(server);
			}
		});
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
			try {
				Train.shutdown(server);
			} catch (Throwable t) {
				System.err.println("[craftbot-train] shutdown: " + t);
			}
			for (Bot bot : new ArrayList<>(BOTS.values())) drop(server, bot);
		});
	}

	/** Runs a /aibot train subcommand; any exception becomes a message to the sender, never a server crash. */
	private static int trainCmd(CommandContext<CommandSourceStack> c, Runnable action) {
		try {
			action.run();
			return 1;
		} catch (Exception e) {
			c.getSource().sendFailure(Component.literal("train: " + e));
			return 0;
		}
	}

	private static int spawn(CommandContext<CommandSourceStack> c, String name) {
		CommandSourceStack src = c.getSource();
		MinecraftServer server = src.getServer();
		if (BOTS.containsKey(name) || server.getPlayerList().getPlayerByName(name) != null) {
			src.sendFailure(Component.literal("a player named " + name + " already exists"));
			return 0;
		}
		ServerLevel level = src.getLevel();
		Vec3 pos = src.getPosition();
		float yaw = src.getRotation().y;
		if (src.getEntity() instanceof ServerPlayer runner) {
			// Two blocks in front of the player who ran the command, turned to face them.
			double[] d = Obs.lookDir(runner.getYRot(), 0f);  // horizontal look vector
			pos = new Vec3(runner.getX() + 2 * d[0], runner.getY(), runner.getZ() + 2 * d[2]);
			yaw = runner.getYRot() + 180f;
			level = (ServerLevel) runner.level();
		}

		GameProfile profile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(name), name);
		ServerPlayer player = new ServerPlayer(server, level, profile, ClientInformation.createDefault());
		player.snapTo(pos.x, pos.y, pos.z, yaw, 0f);
		server.getPlayerList().placeNewPlayer(new FakeConnection(), player,
			new CommonListenerCookie(profile, 0, player.clientInformation(), false));
		player.setGameMode(GameType.SURVIVAL);
		player.setHealth(20f);

		Bot bot = new Bot(name, player, POLICY);
		BOTS.put(name, bot);
		src.sendSuccess(() -> Component.literal("spawned " + name + " (paused). Give it a goal: /aibot task log"), true);
		return 1;
	}

	private static int task(CommandContext<CommandSourceStack> c, String only) {
		String item = StringArgumentType.getString(c, "item");
		if (!Arrays.asList(Obs.ITEM_NAMES).contains(item)) {
			c.getSource().sendFailure(Component.literal("unknown item. Options: " + String.join(", ", Obs.ITEM_NAMES)));
			return 0;
		}
		if (BOTS.isEmpty()) {
			c.getSource().sendFailure(Component.literal("no bots - /aibot spawn first"));
			return 0;
		}
		List<Bot> targets = new ArrayList<>(BOTS.values());
		if (only != null) {
			Bot bot = BOTS.get(only);
			if (bot == null) {
				c.getSource().sendFailure(Component.literal("no bot named " + only));
				return 0;
			}
			targets = List.of(bot);
		}
		for (Bot bot : targets) bot.startTask(c.getSource().getServer(), item);
		return targets.size();
	}

	/** Debug: pause the policy and run one hand-picked action on every bot. */
	private static int act(CommandContext<CommandSourceStack> c) {
		int a = Arrays.asList(Bot.ACTIONS).indexOf(StringArgumentType.getString(c, "action"));
		if (a < 0) {
			c.getSource().sendFailure(Component.literal("unknown action"));
			return 0;
		}
		for (Bot bot : BOTS.values()) bot.force(a);
		return 1;
	}

	/** /aibot dump <bot>: parity dump (format minemind-dump-1) into <game dir>/minemind_dumps/. */
	private static int dump(CommandContext<CommandSourceStack> c) {
		String name = StringArgumentType.getString(c, "name");
		Bot bot = BOTS.get(name);
		if (bot == null) {
			c.getSource().sendFailure(Component.literal("no bot named " + name));
			return 0;
		}
		try {
			java.nio.file.Path file = bot.dump(c.getSource().getServer(), Bot.dumpDir());
			c.getSource().sendSuccess(() -> Component.literal("dumped " + name + " -> " + file), true);
			return 1;
		} catch (java.io.IOException e) {
			c.getSource().sendFailure(Component.literal("dump failed: " + e));
			return 0;
		}
	}

	private static int setPaused(CommandContext<CommandSourceStack> c, boolean paused) {
		for (Bot bot : BOTS.values()) bot.paused = paused;
		c.getSource().sendSuccess(() -> Component.literal((paused ? "paused " : "resumed ") + BOTS.size() + " bot(s)"), true);
		return BOTS.size();
	}

	private static int status(CommandContext<CommandSourceStack> c) {
		if (BOTS.isEmpty()) c.getSource().sendSuccess(() -> Component.literal("no bots"), false);
		for (Bot bot : BOTS.values()) {
			String s = bot.status();
			c.getSource().sendSuccess(() -> Component.literal(s), false);
		}
		return BOTS.size();
	}

	private static int remove(CommandContext<CommandSourceStack> c, String name) {
		MinecraftServer server = c.getSource().getServer();
		List<Bot> targets = new ArrayList<>();
		for (Bot bot : BOTS.values()) if (name == null || bot.name.equals(name)) targets.add(bot);
		for (Bot bot : targets) drop(server, bot);
		c.getSource().sendSuccess(() -> Component.literal("removed " + targets.size() + " bot(s)"), true);
		return targets.size();
	}

	private static void drop(MinecraftServer server, Bot bot) {
		BOTS.remove(bot.name);
		if (server.getPlayerList().getPlayer(bot.player.getUUID()) != null) server.getPlayerList().remove(bot.player);
	}
}
