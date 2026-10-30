package com.minemind.craftbot;

import io.netty.channel.ChannelFutureListener;
import net.minecraft.network.Connection;
import net.minecraft.network.DisconnectionDetails;
import net.minecraft.network.PacketListener;
import net.minecraft.network.ProtocolInfo;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import org.jspecify.annotations.Nullable;

/**
 * A connection with no socket behind it: every outgoing packet is dropped. It is never registered
 * with the server's connection list, so nothing ticks it (no keep-alive timeouts).
 */
public class FakeConnection extends Connection {
	public FakeConnection() {
		super(PacketFlow.SERVERBOUND);
	}

	@Override
	public void send(final Packet<?> packet, final @Nullable ChannelFutureListener listener, final boolean flush) {
	}

	@Override
	public <T extends PacketListener> void setupInboundProtocol(final ProtocolInfo<T> protocol, final T packetListener) {
	}

	@Override
	public void setupOutboundProtocol(final ProtocolInfo<?> protocol) {
	}

	@Override
	public void flushChannel() {
	}

	@Override
	public void disconnect(final DisconnectionDetails details) {
	}

	@Override
	public void setReadOnly() {
	}

	@Override
	public void handleDisconnection() {
	}
}
