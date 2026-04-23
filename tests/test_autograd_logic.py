"""Integration test for remote forward + backward passes.

This test spins up a server in a background thread, connects a client, and
verifies that gradients flow correctly through the remote module.

Prerequisites:
    - Hivemind DHT must be reachable (or the test runs locally with ``use_ipfs``).
    - The test is designed for manual / CI runs where P2P networking is available.

NOTE: In environments without P2P connectivity this test will hang or fail.
      It is therefore not included in the default ``pytest`` discovery — run it
      explicitly with ``pytest -v tests/test_autograd_logic.py``.
"""

import asyncio
import threading

import torch
from hivemind import DHT
from hivemind.p2p import PeerID

from p2p_module_calling.client import Client, RemoteModule
from p2p_module_calling.server import ModuleServicer


class TestModel(torch.nn.Module):
    """Simple linear model: y = Wx + b, returning ``{"result": y}``."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> dict:
        return {"result": self.linear(x)}


def _server_main(dht: DHT, ready_event: threading.Event) -> None:
    """Run the servicer in a background thread."""
    model = TestModel()
    models = {"test_model": model}
    servicer = ModuleServicer(dht, models)
    servicer.run_in_background()
    ready_event.set()
    # Block until the test is done (the thread is daemonised so it won't
    # prevent process exit).
    servicer.join()


def test_remote_forward_backward():
    """End-to-end test: forward + backward through a remote module."""
    dht = DHT(
        start=False,
        use_auto_relay=True,
        use_relay=True,
        use_ipfs=True,
    )
    dht.run_in_background(timeout=60)

    # Start the server in a background thread and wait for it to be ready.
    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=_server_main,
        args=(dht, ready_event),
        daemon=True,
    )
    server_thread.start()
    ready_event.wait(timeout=30)

    server_peer_id = dht.peer_id

    with Client(dht, server_peer_id) as client:
        remote_module = RemoteModule(client, "test_model")

        with torch.enable_grad():
            x = torch.ones(2, 10, requires_grad=True)
            result = remote_module({"x": x})
            assert "result" in result
            result["result"].sum().backward()
            assert x.grad is not None, "Gradient should have been computed"
            assert x.grad.shape == x.shape, f"Grad shape mismatch: {x.grad.shape} vs {x.shape}"


if __name__ == "__main__":
    test_remote_forward_backward()
    print("Test passed!")
