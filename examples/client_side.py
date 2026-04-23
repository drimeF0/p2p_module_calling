"""Example: client side — call a remote module with autograd support.

Run this on a different machine (or in a different terminal).  You will be
prompted to enter the server's Peer ID printed by ``server_side.py``.
"""

import torch
from hivemind import DHT
from hivemind.p2p import PeerID

from p2p_module_calling.client import Client, RemoteModule


def main() -> None:
    dht = DHT(
        start=False,
        use_auto_relay=True,
        use_relay=True,
        use_ipfs=True,
    )
    dht.run_in_background(timeout=60)

    server_peer_id = PeerID.from_base58(input("Enter server peer ID: "))

    with Client(dht, server_peer_id) as client:
        remote_module = RemoteModule(client, "test_model")

        with torch.enable_grad():
            data = {"x": torch.ones(1, 10, requires_grad=True)}
            result = remote_module(data)
            print(f"Forward result: {result}")
            result["result"].sum().backward()
            print(f"Input gradient: {data['x'].grad}")


if __name__ == "__main__":
    main()
