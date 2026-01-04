import os
import time
import asyncio
import argparse
from collections import deque
import websockets

# Efficiently read last n lines from a file
def last_lines(fname, n=10, chunk=1024):
    with open(fname, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size, data, lines = f.tell(), b'', deque()
        while size > 0 and len(lines) <= n:
            read_size = min(chunk, size)
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
            lines = deque(data.split(b'\n'), maxlen=n + 1)
        return [l.decode(errors='ignore') for l in list(lines)[-n:]]


# Tail function with WebSocket streaming
async def tail_and_stream(websocket, path, fname, n=10, follow=False, interval=1):
    try:
        # Send last n lines first
        for line in last_lines(fname, n):
            await websocket.send(line)

        if follow:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, os.SEEK_END)
                pos = f.tell()

                while True:
                    size = os.path.getsize(fname)
                    if size < pos:  # File truncated
                        await websocket.send("==> File truncated <==")
                        f.seek(0)
                        pos = 0

                    elif size > pos:  # New lines added
                        f.seek(pos)
                        for line in f.read(size - pos).splitlines():
                            await websocket.send(line)
                        pos = f.tell()

                    await asyncio.sleep(interval)

    except Exception as e:
        print(f"Error: {e}")


# Main WebSocket server
async def main(filename, n, port, follow, interval):
    async def handler(websocket):
        await tail_and_stream(websocket, None, filename, n, follow, interval)

    async with websockets.serve(handler, "localhost", port):
        print(f"WebSocket server running on ws://localhost:{port}")
        await asyncio.Future()  # Run forever



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tail log file via WebSocket")
    parser.add_argument("filename", help="Log file to watch")
    parser.add_argument("-n", type=int, default=10, help="Number of last lines to send")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow the file (like tail -f)")
    parser.add_argument("-s", "--sleep", type=float, default=1, help="Polling interval in seconds")
    parser.add_argument("-p", "--port", type=int, default=8100, help="WebSocket port (default 8100)")
    args = parser.parse_args()

    asyncio.run(main(args.filename, args.n, args.port, args.follow, args.sleep))










# For terminal

# import os, time, sys, argparse
# from collections import deque
#
# def last_lines(fname, n=10, chunk=1024):
#     with open(fname, 'rb') as f:
#         f.seek(0, os.SEEK_END)
#         size, data, lines = f.tell(), b'', deque()
#         while size > 0 and len(lines) <= n:
#             read_size = min(chunk, size)
#             size -= read_size
#             f.seek(size)
#             data = f.read(read_size) + data
#             lines = deque(data.split(b'\n'), maxlen=n+1)
#         return [l.decode(errors='ignore') for l in list(lines)[-n:]]
#
# def tail(fname, n=10, follow=False, interval=1):
#     try:
#         for line in last_lines(fname, n):
#             print(line)
#
#         with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
#             f.seek(0, os.SEEK_END)
#             pos = f.tell()
#
#             while follow:
#                 size = os.path.getsize(fname)
#                 if size < pos:  # file truncated
#                     print("==> File truncated <==")
#                     f.seek(0)
#                     pos = 0
#
#                 elif size > pos:
#                     f.seek(pos)
#                     for line in f.read(size - pos).splitlines():
#                         print(line)
#                     pos = f.tell()
#
#                 time.sleep(interval)
#
#     except FileNotFoundError:
#         print(f"tail: {fname}: No such file")
#         sys.exit(1)
#     except KeyboardInterrupt:
#         sys.exit(0)
#
# if __name__ == "__main__":
#     p = argparse.ArgumentParser(description="Hybrid tail -f")
#     p.add_argument("filename")
#     p.add_argument("-n", type=int, default=10, help="Number of lines (default 10)")
#     p.add_argument("-f", "--follow", action="store_true", help="Follow file")
#     p.add_argument("-s", "--sleep", type=float, default=1, help="Sleep interval")
#     a = p.parse_args()
#
#     tail(a.filename, a.n, a.follow, a.sleep)