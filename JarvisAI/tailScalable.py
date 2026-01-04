import asyncio
import argparse
import os
import websockets
from collections import deque

# Store all connected clients
clients = set()

async def tail_file(filepath, num_lines, sleep_sec):
    """Async generator to yield new lines from a file, handling truncation."""
    with open(filepath, "r") as f:
        # Show last n lines
        dq = deque(f, maxlen=num_lines)
        for line in dq:
            yield line.rstrip("\n")

        # Seek to EOF
        f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if line:
                yield line.rstrip("\n")
            else:
                await asyncio.sleep(sleep_sec)

async def broadcast(filepath, num_lines, sleep_sec):
    """Continuously read the file and send updates to all clients."""
    async for line in tail_file(filepath, num_lines, sleep_sec):
        if clients:  # Only broadcast if clients are connected
            await asyncio.gather(
                *(client.send(line) for client in clients if client.open),
                return_exceptions=True
            )

async def handler(websocket, path):  # <-- FIX: added path here
    """Handle new WebSocket connections."""
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def main(filepath, num_lines, sleep_sec, host="localhost", port=8100):
    print(f"🚀 WebSocket server running on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await broadcast(filepath, num_lines, sleep_sec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail a file over WebSocket")
    parser.add_argument("filepath", help="File to tail")
    parser.add_argument("-n", "--lines", type=int, default=10, help="Number of lines to show")
    parser.add_argument("-s", "--sleep", type=float, default=1.0, help="Polling interval in seconds")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.filepath, args.lines, args.sleep))
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
