package com.minemind.recorder;

import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.entity.event.v1.EntitySleepEvents;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.entity.event.v1.ServerPlayerEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.AttackEntityCallback;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.fabricmc.fabric.api.message.v1.ServerMessageEvents;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.commands.SharedSuggestionProvider;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionResult;

import java.util.ArrayList;
import java.util.List;

/**
 * Server side of the recorder (PROTOCOL 9): /aibot record start|stop|pause|resume|status, /aibot context [text|finish],
 * and the event hooks. Commands share the "aibot" root with craftbot; Brigadier merges the literal trees.
 */
public final class RecorderMod implements ModInitializer {
	@Override
	public void onInitialize() {
		Recorder.init();

		CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> dispatcher.register(
			Commands.literal("aibot")
				.then(Commands.literal("record")
					.then(Commands.literal("start")
						.executes(c -> start(c, "none"))
						.then(Commands.argument("goal", StringArgumentType.word())
							.suggests((c, b) -> SharedSuggestionProvider.suggest(goalOptions(), b))
							.executes(c -> start(c, StringArgumentType.getString(c, "goal")))))
					.then(Commands.literal("stop").executes(c -> reply(c, Recorder.stop("command"))))
					.then(Commands.literal("pause").executes(c -> reply(c, Recorder.pause())))
					.then(Commands.literal("resume").executes(c -> reply(c, Recorder.resume())))
					.then(Commands.literal("status").executes(c -> reply(c, Recorder.status()))))
				.then(Commands.literal("context")
					.executes(c -> reply(c, Recorder.contextShow()))
					.then(Commands.literal("finish").executes(c -> reply(c, Recorder.contextFinish())))
					.then(Commands.argument("text", StringArgumentType.greedyString())
						.executes(c -> reply(c, Recorder.contextSet(StringArgumentType.getString(c, "text"))))))));

		ServerTickEvents.END_SERVER_TICK.register(Recorder::tick);
		ServerLifecycleEvents.SERVER_STOPPING.register(server -> Recorder.serverStopping());
		ServerPlayConnectionEvents.JOIN.register((handler, sender, server) ->
			Recorder.onJoin(handler.getPlayer().getUUID()));
		ServerPlayConnectionEvents.DISCONNECT.register((handler, server) ->
			Recorder.onDisconnect(handler.getPlayer().getUUID()));
		PlayerBlockBreakEvents.AFTER.register((level, player, pos, state, be) ->
			Recorder.onBreak(level, player, pos, state));
		AttackEntityCallback.EVENT.register((player, level, hand, entity, hit) -> {
			Recorder.onAttack(player, entity);
			return InteractionResult.PASS;
		});
		ServerLivingEntityEvents.ALLOW_DAMAGE.register((entity, source, amount) -> {
			Recorder.onAllowDamage(entity, source);
			return true;
		});
		ServerLivingEntityEvents.AFTER_DEATH.register(Recorder::onAfterDeath);
		ServerPlayerEvents.AFTER_RESPAWN.register((oldPlayer, newPlayer, alive) -> Recorder.onRespawn(newPlayer));
		EntitySleepEvents.START_SLEEPING.register((entity, pos) -> Recorder.onSleep(entity, pos, true));
		EntitySleepEvents.STOP_SLEEPING.register((entity, pos) -> Recorder.onSleep(entity, pos, false));
		ServerMessageEvents.CHAT_MESSAGE.register((message, sender, bound) ->
			Recorder.onChat(sender, message.signedContent()));
	}

	static List<String> goalOptions() {
		List<String> out = new ArrayList<>(List.of(Obs.ITEM_NAMES));
		out.add("none");
		return out;
	}

	private static int start(CommandContext<CommandSourceStack> c, String goal) {
		CommandSourceStack src = c.getSource();
		MinecraftServer server = src.getServer();
		ServerPlayer player = src.getPlayer();
		if (player == null && !server.getPlayerList().getPlayers().isEmpty()) player = server.getPlayerList().getPlayers().get(0);
		if (player == null) {
			src.sendFailure(Component.literal("no player to record"));
			return 0;
		}
		String msg = Recorder.start(server, player, goal);
		boolean ok = msg.startsWith("recording to");
		if (ok) src.sendSuccess(() -> Component.literal(msg), true);
		else src.sendFailure(Component.literal(msg));
		return ok ? 1 : 0;
	}

	private static int reply(CommandContext<CommandSourceStack> c, String msg) {
		c.getSource().sendSuccess(() -> Component.literal(msg), false);
		return 1;
	}
}
