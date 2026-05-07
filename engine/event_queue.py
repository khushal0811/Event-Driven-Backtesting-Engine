"""
event_queue.py — FIFO event queue for the backtesting engine.

Wraps collections.deque to enforce strict FIFO ordering.
The engine appends to the right and consumes from the left,
guaranteeing deterministic event processing.
"""

from collections import deque
from typing import Optional

from engine.events import Event


class EventQueue:
    """
    Thread-unsafe, single-threaded FIFO queue.

    Usage:
        q = EventQueue()
        q.put(event)
        if not q.empty():
            event = q.get()
    """

    def __init__(self) -> None:
        self._queue: deque[Event] = deque()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def put(self, event: Event) -> None:
        """Append an event to the tail of the queue."""
        self._queue.append(event)

    def get(self) -> Optional[Event]:
        """
        Remove and return the event at the head of the queue.

        Returns None if the queue is empty (non-blocking).
        """
        if self._queue:
            return self._queue.popleft()
        return None

    def empty(self) -> bool:
        """Return True if the queue contains no events."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)

    def __repr__(self) -> str:
        return f"EventQueue(size={len(self._queue)})"
