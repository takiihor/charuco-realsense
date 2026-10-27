"""Signal handling that never raises inside a Tk callback or calls Tcl."""
import signal
import threading


class SignalStop:
    def __init__(self):
        self.requested = threading.Event()
        self.signal_number = None
        self._previous = {}

    def _handle(self, number, frame):
        if self.signal_number is None:
            self.signal_number = number
        self.requested.set()

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.signal(number, self._handle)
        return self

    def __exit__(self, *exc):
        for number, handler in self._previous.items():
            signal.signal(number, handler)
