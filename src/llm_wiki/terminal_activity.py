"""Append-only activity history with one ephemeral, thread-safe status line."""

import os
import shutil
import sys
import threading
import time
import unicodedata


def clean(text, *, multiline=False):
    return "".join(c for c in str(text) if c.isprintable() or (multiline and c in "\n\t"))


class ActivityLog:
    def __init__(self, *, live=None, clock=time.monotonic, stream=None):
        self.live = (sys.stdout.isatty() and sys.stderr.isatty()
                     and os.environ.get("TERM") != "dumb") if live is None else live
        self.clock = clock
        self.stream = stream if stream is not None else sys.stderr
        self.lock = threading.RLock()
        self.active = {}
        self.worker = None
        self.stop = threading.Event()
        self.visible = False
        self.text_open = False

    def _clear(self):
        if self.visible:
            self.stream.write("\r\x1b[2K")
            self.stream.flush()
            self.visible = False

    def status(self):
        now = self.clock()
        items = []
        for label, started in reversed(list(self.active.values())):
            elapsed = int(now - started)
            items.append(label + (f" · 已等待 {elapsed} 秒" if elapsed >= 1 else ""))
        return " | ".join(items)

    def refresh(self):
        with self.lock:
            if not self.live or self.text_open:
                return
            self._clear()
            if not self.active:
                return
            text = "⠋ " + self.status()
            width = max(1, shutil.get_terminal_size().columns - 1)
            clipped, used = [], 0
            for c in text:
                size = 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
                if used + size > width:
                    break
                clipped.append(c)
                used += size
            self.stream.write("".join(clipped))
            self.stream.flush()
            self.visible = True

    def _tick(self):
        while not self.stop.wait(1):
            self.refresh()

    def begin(self, key, label, *, log=True):
        with self.lock:
            self.active[key] = (clean(label), self.clock())
            if self.live:
                if self.worker is None:
                    self.stop.clear()
                    self.worker = threading.Thread(target=self._tick, daemon=True)
                    self.worker.start()
                self.refresh()
            elif log:
                self.line("› " + label)

    def finish(self, key, message=None):
        with self.lock:
            self.active.pop(key, None)
            if message:
                self.line(message)
            else:
                self.refresh()

    def line(self, message):
        with self.lock:
            self._clear()
            self.end_text()
            print(clean(message, multiline=True), flush=True)
            self.refresh()

    def text(self, fragment):
        with self.lock:
            self._clear()
            fragment = clean(fragment, multiline=True)
            if fragment:
                sys.stdout.write(fragment)
                sys.stdout.flush()
                self.text_open = True

    def end_text(self):
        with self.lock:
            if self.text_open:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.text_open = False

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()
            self.worker = None
        with self.lock:
            self._clear()
            self.end_text()
            self.active.clear()
