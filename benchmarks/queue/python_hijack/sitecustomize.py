import zmq
import os
import sys
import asyncio

# --- ZMQ CRIU PATCH ---
_original_poller = zmq.Poller

class CriuSafePoller(_original_poller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._creator_pid = os.getpid()
        self._registry = {}

    def register(self, socket, flags=zmq.POLLIN):
        self._registry[socket] = flags
        return super().register(socket, flags)

    def unregister(self, socket):
        if socket in self._registry:
            del self._registry[socket]
        return super().unregister(socket)

    def poll(self, timeout=None):
        current_pid = os.getpid()
        if current_pid != self._creator_pid:
            super().__init__()
            for socket, flags in self._registry.items():
                super().register(socket, flags)
            self._creator_pid = current_pid
        # Cap at 1000ms: workers resumed inside poller.poll() at CRIU restore
        # will escape within 1s rather than blocking for VLLM_RINGBUFFER_WARNING_INTERVAL
        # (~60s). After escaping, SpinCondition.wait()→sched_yield() patch takes over.
        capped = min(timeout, 1000) if timeout is not None else 1000
        return super().poll(capped)

zmq.Poller = CriuSafePoller

# --- ASYNCIO CRIU PATCH ---
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# --- SHM BROADCAST CRIU PATCH ---
# SpinCondition.wait() uses zmq.Poller on two sockets:
#   1. read_cancel_socket  — inproc://, DEAD after CRIU (no real fd)
#   2. local_notify_socket — PUB/SUB, may also be broken post-restore
# After restore the system has been idle >> busy_loop_s (1s), so wait()
# always enters idle mode and polls the broken sockets → timeout every time.
#
# Fix: replace wait() with a pure sched_yield().  The caller (acquire_read)
# loops checking the SHM slot metadata (written_flag in shared memory, which
# CRIU does preserve).  sched_yield() gives the writer CPU time to fill a
# slot; the next iteration of acquire_read's loop finds it.

def _install_shm_broadcast_patch():
    import importlib.abc

    _in_find = False  # re-entrancy guard

    class _ShmFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            nonlocal _in_find
            if fullname != 'vllm.distributed.device_communicators.shm_broadcast':
                return None
            if _in_find:
                return None
            _in_find = True
            try:
                import importlib.util
                spec = importlib.util.find_spec(fullname)
            except Exception:
                return None
            finally:
                _in_find = False
            if spec is None:
                return None
            spec.loader = _ShmLoader(spec.loader)
            return spec

    class _ShmLoader(importlib.abc.Loader):
        def __init__(self, real):
            self._real = real

        def create_module(self, spec):
            fn = getattr(self._real, 'create_module', None)
            return fn(spec) if fn else None

        def exec_module(self, module):
            self._real.exec_module(module)
            from os import sched_yield
            SpinCondition = getattr(module, 'SpinCondition', None)
            if SpinCondition is not None:
                SpinCondition.wait = lambda self, timeout_ms=None: sched_yield()

                _orig_notify = getattr(SpinCondition, 'notify', None)
                if _orig_notify is not None:
                    def _safe_notify(self_sc, *a, **kw):
                        try:
                            _orig_notify(self_sc, *a, **kw)
                        except Exception as _e:
                            print(f"[CRIU-PATCH] SpinCondition.notify() suppressed: {_e}",
                                  file=sys.stderr, flush=True)
                    SpinCondition.notify = _safe_notify

                print(
                    f"[CRIU-PATCH] SpinCondition.wait()+notify() patched (pid={os.getpid()})",
                    file=sys.stderr, flush=True,
                )

    sys.meta_path.insert(0, _ShmFinder())

_install_shm_broadcast_patch()

print(f"[CRIU-FIX] sitecustomize loaded in PID {os.getpid()}", file=sys.stderr)
