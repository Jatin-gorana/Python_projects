import os
import time
import sys
from collections import deque


class TailF:
    def __init__(self, filename, lines=10, follow=True, sleep_interval=1):
        self.filename = filename
        self.lines = lines
        self.follow = follow
        self.sleep_interval = sleep_interval
        self.file_size = 0

    def get_last_lines(self, file_handle, num_lines):
        """Get the last N lines from a file efficiently"""
        if num_lines <= 0:
            return []

        # Move to end of file
        file_handle.seek(0, os.SEEK_END)
        file_size = file_handle.tell()

        if file_size == 0:
            return []

        # Read file in chunks from the end
        lines = deque()
        buffer = ""
        chunk_size = 1024
        position = file_size

        while position > 0 and len(lines) < num_lines:
            # Calculate how much to read
            read_size = min(chunk_size, position)
            position -= read_size

            # Read chunk
            file_handle.seek(position)
            chunk = file_handle.read(read_size)

            # Prepend to buffer
            buffer = chunk + buffer

            # Split into lines
            lines_in_buffer = buffer.split('\n')

            # Keep the incomplete line at the beginning for next iteration
            buffer = lines_in_buffer[0]

            # Add complete lines to our deque (in reverse order)
            for line in reversed(lines_in_buffer[1:]):
                if len(lines) < num_lines:
                    lines.appendleft(line)
                else:
                    break

            if position == 0 and buffer:
                lines.appendleft(buffer)

        return list(lines)[-num_lines:]

    def tail_follow(self):
        """Main tail -f functionality"""
        try:
            with open(self.filename, 'r', encoding='utf-8', errors='ignore') as file:
                # Print initial lines
                initial_lines = self.get_last_lines(file, self.lines)
                for line in initial_lines:
                    print(line.rstrip('\n'))

                # Get current file size and position
                file.seek(0, os.SEEK_END)
                self.file_size = file.tell()

                if not self.follow:
                    return

                # Continuous monitoring
                while True:
                    # Check if file has grown
                    current_size = os.path.getsize(self.filename)

                    if current_size > self.file_size:
                        # File has grown, read new content
                        file.seek(self.file_size)
                        new_content = file.read(current_size - self.file_size)

                        # Print new lines
                        for line in new_content.splitlines():
                            print(line)

                        self.file_size = current_size

                    elif current_size < self.file_size:
                        # File has been truncated or rotated
                        print("==> File truncated <==")
                        file.seek(0)
                        self.file_size = 0

                    time.sleep(self.sleep_interval)

        except FileNotFoundError:
            print(f"tail: {self.filename}: No such file or directory")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


# Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Mimic tail -f command')
    parser.add_argument('filename', help='File to tail')
    parser.add_argument('-n', '--lines', type=int, default=10,
                        help='Number of lines to show initially (default: 10)')
    parser.add_argument('-f', '--follow', action='store_true', default=True,
                        help='Follow file for new content')
    parser.add_argument('-s', '--sleep-interval', type=float, default=1,
                        help='Sleep interval between checks (default: 1 second)')

    args = parser.parse_args()

    tail = TailF(args.filename, args.lines, args.follow, args.sleep_interval)
    tail.tail_follow()
