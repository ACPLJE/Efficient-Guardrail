"""SPEC 4.1: VirtualClock discrete-event semantics."""
import threading

from ag.harness.context import VirtualClock


def test_single_thread_jumps():
    c = VirtualClock()
    c.sleep_until(100)
    assert c.now_ns() == 100
    c.sleep_until(50)
    assert c.now_ns() == 100
    c.advance(5)
    assert c.now_ns() == 105


def test_registered_threads_wake_in_time_order():
    c = VirtualClock()
    order: list[tuple[str, int]] = []
    lock = threading.Lock()
    ready = threading.Barrier(2)

    def worker(name, times):
        c.register()
        ready.wait()
        for t in times:
            c.sleep_until(t)
            with lock:
                order.append((name, c.now_ns()))
        c.unregister()

    a = threading.Thread(target=worker, args=("a", [10, 30]))
    b = threading.Thread(target=worker, args=("b", [20, 40]))
    a.start(); b.start(); a.join(5); b.join(5)
    assert not a.is_alive() and not b.is_alive()
    assert order == [("a", 10), ("b", 20), ("a", 30), ("b", 40)]
