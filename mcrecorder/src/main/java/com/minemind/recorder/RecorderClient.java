package com.minemind.recorder;

import com.mojang.blaze3d.platform.InputConstants;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keymapping.v1.KeyMappingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;
import org.lwjgl.glfw.GLFW;

/** Client side: F8 starts / pauses / resumes (singleplayer), raw input is sampled every client tick, HUD. */
public final class RecorderClient implements ClientModInitializer {
	private static KeyMapping toggle;
	private static final Identifier HUD_ID = Identifier.fromNamespaceAndPath("minemind_recorder", "rec_hud");

	@Override
	public void onInitializeClient() {
		toggle = KeyMappingHelper.registerKeyMapping(new KeyMapping("key.minemind_recorder.toggle",
			InputConstants.Type.KEYSYM, GLFW.GLFW_KEY_F8, KeyMapping.Category.MISC));

		ClientTickEvents.END_CLIENT_TICK.register(mc -> {
			try {
				InputTap.onClientTick(mc);
				while (toggle.consumeClick()) {
					MinecraftServer server = mc.getSingleplayerServer();
					if (server != null) server.execute(() -> {
						try {
							Recorder.toggleFromKey(server);
						} catch (Throwable t) {
							CraftLog.error("F8 toggle failed", t);
						}
					});
				}
			} catch (Throwable t) {
				CraftLog.error("client tick failed", t);
			}
		});

		HudElementRegistry.addLast(HUD_ID, (graphics, delta) -> {
			try {
				String status = Recorder.hudStatus();
				if (status == null) return;
				Minecraft mc = Minecraft.getInstance();
				boolean paused = Recorder.paused;
				graphics.fill(4, 4, 12, 12, paused ? 0xFFD0A020 : 0xFFD02020);
				graphics.text(mc.font, status, 16, 5, paused ? 0xFFFFD040 : 0xFFFF5555);
				String ctx = Recorder.hudContext();
				if (ctx != null) graphics.text(mc.font, "context: " + ctx, 16, 16, 0xFFFFFFFF);
			} catch (Throwable t) {
				CraftLog.error("hud failed", t);
			}
		});
	}
}
