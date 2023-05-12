from typing import Callable


class NPOutputReader:
    def __init__(self, callback: Callable[[str], None]):
        self.buffer = ""
        self.onLineRead = callback

    def flush(self):
        pass

    def close(self):
        pass

    def write(self, message: str):
        message = message.replace("\r", "\n")
        if "\n" not in message:
            self.buffer += message
        else:
            parts = message.split("\n")
            if self.buffer:
                s = self.buffer + parts[0]
                self.onLineRead(s)
            self.buffer = parts.pop()
            for part in parts:
                self.onLineRead(part)
