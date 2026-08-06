#!/usr/bin/env python3
import asyncio
import json
import nats

async def test():
    try:
        nc = await nats.connect("nats://nats:4222")
        msg = {"test": "ZALUPUPA!"}
        await nc.publish("vision.exchange", json.dumps(msg).encode())
        await nc.drain()
        print("✓ Test message sent to vision.exchange")
    except Exception as e:
        print(f"✗ Error: {e}")

asyncio.run(test())
