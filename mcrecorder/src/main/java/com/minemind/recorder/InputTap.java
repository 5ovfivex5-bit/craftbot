package com.minemind.recorder;

import net.minecraft.client.Minecraft;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.inventory.AbstractFurnaceScreen;
import net.minecraft.client.gui.screens.inventory.ContainerScreen;
import net.minecraft.client.gui.screens.inventory.CraftingScreen;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.phys.BlockHitResult;

/** Client side: reads the raw input state once per client tick (keys, mouse-look delta, open screen). */
public final class InputTap {
	private InputTap() {}

	private static boolean havePrev;
	private static float prevYaw, prevPitch;

	public static void onClientTick(Minecraft mc) {
		if (!InputSamples.enabled) {
			havePrev = false;
			return;
		}
		LocalPlayer p = mc.player;
		ClientLevel level = mc.level;
		if (p == null || level == null) {
			havePrev = false;
			return;
		}
		Options o = mc.options;
		InputSamples.Sample s = new InputSamples.Sample();
		s.forward = o.keyUp.isDown();
		s.back = o.keyDown.isDown();
		s.left = o.keyLeft.isDown();
		s.right = o.keyRight.isDown();
		s.jump = o.keyJump.isDown();
		s.sneak = o.keyShift.isDown();
		s.sprint = o.keySprint.isDown();
		s.attack = o.keyAttack.isDown();
		s.use = o.keyUse.isDown();
		s.drop = o.keyDrop.isDown();
		s.inventory = o.keyInventory.isDown();
		s.swapOffhand = o.keySwapOffhand.isDown();
		s.pick = o.keyPickItem.isDown();
		for (int i = 0; i < 9; i++) s.hotbar[i] = o.keyHotbarSlots[i].isDown();

		float yaw = p.getYRot(), pitch = p.getXRot();
		if (havePrev) {
			s.dyaw = wrap(yaw - prevYaw);
			s.dpitch = pitch - prevPitch;
		}
		havePrev = true;
		prevYaw = yaw;
		prevPitch = pitch;
		s.yaw = yaw;
		s.pitch = pitch;
		s.selectedSlot = p.getInventory().getSelectedSlot();
		s.attackBlock = s.attack && mc.hitResult instanceof BlockHitResult;
		s.gui = guiOf(mc.gui.screen());
		InputSamples.put(s);
	}

	static float wrap(float d) {
		d = ((d + 180f) % 360f + 360f) % 360f - 180f;
		return d;
	}

	static String guiOf(Screen sc) {
		if (sc == null) return null;
		if (sc instanceof InventoryScreen) return "inventory";
		if (sc instanceof CraftingScreen) return "crafting";
		if (sc instanceof AbstractFurnaceScreen<?>) return "furnace";
		if (sc instanceof ContainerScreen) return "chest";
		return sc.getClass().getSimpleName();
	}
}
