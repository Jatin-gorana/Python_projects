import argparse
import os
import websockets
import argparse
import asyncio
from collections import deque

def last_lines(fname, n=10, chunk=1024):
    with open(fname, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size, data, lines = f.tell(), b'', deque()
        while size>0 and len(lines)<=n:
            read_size = min(chunk, size)
            size -= read_size
            f.seek(size)
            data = f.read(read_size) + data
            lines = deque(data.split(b'\n'), maxlen=n)

        return [l.decode(errors='ignore') for l in list(lines)[-n:] if l.strip()]

async def tail_and_stream(websocket, fname, n=10, follow=False, interval=1):
    try:
        for line in last_lines(fname, n):
            await websocket.send(line)

        if (follow):
            try:
                with open(fname, 'r', encoding="utf-8", errors="ignore") as f:
                    f.seek(0, os.SEEK_END)   #responsible for moving file cursor to end
                    pos = f.tell()   #confirms cursor pos

                    while True:
                        size = os.path.getsize(fname)
                        #checking if current file size is greater than prev size where cursor pointed
                        if (size > pos):
                            f.seek(pos)
                            for line in f.read(size-pos).splitlines():
                                await websocket.send(line)

                            pos = f.tell()

                        await asyncio.sleep(interval)

            except FileNotFoundError as e:
                print(e)
                await websocket.send(e)

    except Exception as e:
        print(f"Error: {e}")
        await websocket.send(f"Error: {type(e).__name__} - {str(e)}")


async def main(fname, n, port, follow, interval):
    async def handler(websocket):
        await tail_and_stream(websocket, fname, n, follow, interval)

    async with websockets.serve(handler, "localhost", port):
        print(f"Websocket is running at ws://localhost:{port} ")
        await asyncio.Future()    #this will make it run forever untill exited


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log -f command implementation")
    parser.add_argument("filename", help="Name of file to be logged")
    parser.add_argument("-n", type=int, default=10, help="No. of lines to be logged")
    parser.add_argument("-f", "--follow", action="store_true", help="To follow file in realtime")
    parser.add_argument("-s", "--sleep", type=float, default=1, help="Polling interval")
    parser.add_argument("-p", "--port", type=int, default=8100, help="Port on which frontend should run")

    args = parser.parse_args()

    asyncio.run(main(args.filename, args.n, args.port, args.follow, args.sleep))