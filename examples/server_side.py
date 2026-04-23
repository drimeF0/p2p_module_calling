"""Example: server side — host a simple linear model over P2P.

Run this script on the machine that will serve the model.  It will print the
server's Peer ID (which the client needs) and then block, waiting for requests.
"""

import torch
from hivemind import DHT

from p2p_module_calling.server import ModuleServicer


class TestModel(torch.nn.Module):
    """A trivial model for demonstration purposes."""

    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> dict:
        return {"result": self.mlp(x)}


def main() -> None:
    dht = DHT(
        start=False,
        use_auto_relay=True,
        use_relay=True,
        use_ipfs=True,
    )
    dht.run_in_background(timeout=60)

    my_peer_id = dht.peer_id
    print(f"Server Peer ID: {my_peer_id.to_string()}")
    print(f"Visible addresses: {dht.get_visible_maddrs()}")

    model = TestModel()
    models = {"test_model": model}

    # Quick local sanity check — verify the model works before serving.
    with torch.enable_grad():
        data = {"x": torch.ones(1, 10, requires_grad=True)}
        result = model(**data)
        result["result"].sum().backward()
        print(f"Local test — input grad shape: {data['x'].grad.shape}")

    servicer = ModuleServicer(dht, models)
    servicer.run_in_background()
    print("Servicer is running. Press Ctrl+C to stop.")
    try:
        servicer.join()
    except KeyboardInterrupt:
        servicer.shutdown()


if __name__ == "__main__":
    main()