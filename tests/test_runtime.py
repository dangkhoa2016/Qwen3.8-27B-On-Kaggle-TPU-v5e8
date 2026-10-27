import asyncio
import threading
import pytest
from api.runtime import InferenceRuntime, BackendUnavailable, BackendBusy, State

class FakeRunner:
    def __init__(self):
        self.closed = False
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
    def generate_text(self, prompt, *, max_decode_steps):
        self.calls += 1
        if prompt == "wait":
            self.entered.set()
            self.release.wait(5)
        return {"generated_text": prompt}
    def generate_vision(self, image_path, prompt, *, max_decode_steps):
        self.calls += 1
        return {"generated_text": prompt, "image": image_path}
    def close(self):
        self.closed = True

def test_singleton_lifecycle():
    async def run():
        runner = FakeRunner()
        manager = InferenceRuntime(lambda: runner)
        with pytest.raises(BackendUnavailable):
            await manager.invoke("text", "hello", max_decode_steps=2)
        await asyncio.gather(manager.startup(), manager.startup())
        assert manager.load_count == 1 and manager.state is State.READY
        assert (await manager.invoke("text", "hello", max_decode_steps=2))["generated_text"] == "hello"
        assert (await manager.invoke("vision", "describe", image_path="/tmp/a.png", max_decode_steps=2))["image"] == "/tmp/a.png"
        assert manager.request_count == 2
        await manager.shutdown()
        assert runner.closed and manager.state is State.CLOSED
    asyncio.run(run())

def test_fail_closed_load():
    async def run():
        def factory():
            raise ValueError("no checkpoint")
        manager = InferenceRuntime(factory)
        with pytest.raises(BackendUnavailable):
            await manager.startup()
        assert manager.state is State.FAILED and manager.load_count == 0
    asyncio.run(run())

def test_reject_concurrent_requests():
    async def run():
        runner = FakeRunner()
        manager = InferenceRuntime(lambda: runner)
        await manager.startup()
        first = asyncio.create_task(manager.invoke("text", "wait", max_decode_steps=2))
        assert await asyncio.to_thread(runner.entered.wait, 3)
        with pytest.raises(BackendBusy):
            await manager.invoke("text", "second", max_decode_steps=2)
        runner.release.set()
        await first
        assert runner.calls == 1
        await manager.shutdown()
    asyncio.run(run())
