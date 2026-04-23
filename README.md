# p2p_module_calling

Peer-to-peer remote PyTorch module execution with full autograd support.

This library lets you call a `torch.nn.Module` running on a different machine
as if it were local — gradients flow back transparently through the P2P network
thanks to a custom `torch.autograd.Function`.

## Architecture

| Component | Description |
|---|---|
| **Server** (`ModuleServicer`) | Hosts PyTorch modules in a subprocess, exposes P2P RPC endpoints for forward/backward. |
| **Client** (`Client` / `RemoteModule`) | Connects to a remote server. `RemoteModule` wraps the RPC as a `torch.nn.Module` with autograd. |
| **Protocol** (`module_service_proto.proto`) | Protobuf messages compiled via `betterproto`. |
| **RemoteModuleWorker** | Background `uvloop` thread bridging sync → async for the P2P stack. |

## Installation

```bash
pip install -e .
```

Or, with the raw requirements:

```bash
pip install -r requirements.txt
```

Then compile the protobuf:

```bash
protoc --python_betterproto_out=module_service module_service_proto.proto
```

## Quick Start

### 1. Server — host a model

```python
import torch
from hivemind import DHT
from p2p_module_calling.server import ModuleServicer

model = torch.nn.Linear(10, 10)

dht = DHT(start=False, use_auto_relay=True, use_relay=True, use_ipfs=True)
dht.run_in_background(timeout=60)

print(f"My Peer ID: {dht.peer_id.to_string()}")

servicer = ModuleServicer(dht, {"my_model": model})
servicer.run_in_background()
servicer.join()  # blocks
```

### 2. Client — call it remotely

```python
import torch
from hivemind import DHT
from hivemind.p2p import PeerID
from p2p_module_calling.client import Client, RemoteModule

dht = DHT(start=False, use_auto_relay=True, use_relay=True, use_ipfs=True)
dht.run_in_background(timeout=60)

server_peer_id = PeerID.from_base58("<SERVER_PEER_ID>")

with Client(dht, server_peer_id) as client:
    remote = RemoteModule(client, "my_model")
    x = torch.ones(1, 10, requires_grad=True)
    y = remote({"x": x})
    y["result"].sum().backward()
    print(x.grad)  # gradient flows back from the remote server!
```

## Configuration

Set initial peers for DHT bootstrap via the `P2P_INITIAL_PEERS` environment variable:

```bash
export P2P_INITIAL_PEERS="/ip4/1.2.3.4/tcp/31337/p2p/QmFooBar"
```

## Running Tests

```bash
pytest tests/ -v
```

> **Note:** Integration tests require P2P networking. They may hang or fail in
> isolated environments without connectivity.

## Project Structure

```
p2p_module_calling/
├── client/              # Client, RemoteModule, RemoteModuleWorker
├── server/              # ModuleServicer
├── module_service/      # Auto-generated protobuf stubs (betterproto)
├── utils/               # Tensor serialization, byte splitting, dict helpers
├── constants.py         # Shared configuration (initial peers, etc.)
├── examples/            # Standalone example scripts
└── tests/               # Integration tests
```
