import asyncio
import json
import logging
import os

import nats


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def message_handler(msg):
    subject = msg.subject
    payload = msg.data.decode("utf-8", errors="replace")
    logging.info(f"✓ Received on {subject}: {payload}")

    try:
        parsed = json.loads(payload)
        logging.info(f"  Parsed JSON keys: {list(parsed.keys())}")
    except json.JSONDecodeError:
        pass


async def main():
    nats_url = os.getenv("NATS_URL", "nats://nats:4222")
    vision_subject = os.getenv("VISION_SUBJECT", "vision.results")
    server_subject = os.getenv("SERVER_SUBJECT", "server.messages")

    while True:
        try:
            logging.info(f"Connecting to NATS at {nats_url}")
            nc = await nats.connect(nats_url)

            # Subscribe to both subjects with callbacks
            sub_vision = await nc.subscribe(vision_subject, cb=message_handler)
            sub_server = await nc.subscribe(server_subject, cb=message_handler)

            logging.info(f"✓ Comparator started. Listening on {vision_subject} and {server_subject}")
            
            # Keep the connection alive indefinitely
            # The callbacks will process messages as they arrive
            while True:
                await asyncio.sleep(1)
            
        except Exception as exc:
            logging.exception(f"Comparator error: {exc}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())