"""Process targets for hammering a multiprocessing ReplayBuffer (importable, so they also work with ``spawn``)."""

from __future__ import annotations

import time

import torch

#: Small, so that the writer crosses the "buffer becomes full" transition as often as possible.
HAMMER_CAPACITY = 8


def hammer_writer(rb, stop_event, duration_s: float) -> None:
    """
    Repeatedly clear the buffer and refill it past its capacity. Physical slot ``p`` always holds the
    value ``p``, so a reader's ``get_at(slice(None))`` returns exactly the physical indices it mapped to.
    """
    capacity = rb.capacity
    deadline = time.monotonic() + duration_s
    sizes = (capacity - 1, 2, 1, capacity - 2, 3, capacity, 5)
    try:
        while time.monotonic() < deadline and not stop_event.is_set():
            rb.clear()
            for size in sizes:
                if size >= capacity:
                    # extend() keeps the last `capacity` values and restarts the ring at physical slot 0
                    values = (torch.arange(size) - size) % capacity
                else:
                    values = (torch.arange(size) + (rb.offset + rb.count)) % capacity
                rb.extend(values.to(torch.int64))
    finally:
        stop_event.set()


def hammer_reader(rb, stop_event, result_queue) -> None:
    """Read the whole buffer until the writer stops; report how many reads saw an inconsistent (count, offset)."""
    capacity = rb.capacity
    reads = 0
    bad = []
    while not stop_event.is_set():
        physical = rb.get_at(slice(None)).tolist()
        reads += 1
        count = len(physical)
        if count == 0:
            continue
        # A consistent pair is either (count, 0) or (capacity, offset): logical i -> (i + offset) % count
        offset = physical[0]
        expected = [(i + offset) % count for i in range(count)]
        if physical != expected or (offset != 0 and count != capacity):
            if len(bad) < 5:
                bad.append(physical)
    result_queue.put((reads, bad))
