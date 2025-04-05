from hivemind import DHT, get_dht_time
import time

from p2p_module_calling.server import ModuleServiceServicer
import asyncio

import torch


class MyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, x):
        result = self.linear(x)
        dictonary = {"result": result}
        return dictonary


peers =  [
    # IPv4 DNS addresses
    "/dns/bootstrap1.petals.dev/tcp/31337/p2p/QmedTaZXmULqwspJXz44SsPZyTNKxhnnFvYRajfH7MGhCY",
]

table = DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=peers,
    start=True,
    use_relay=True,
    use_auto_relay=True
)


my_peer_id = table.peer_id.to_bytes()
print(table.peer_id.to_string())
print('\n'.join(str(addr) for addr in table.get_visible_maddrs()))
table.store("drime_peers", my_peer_id, get_dht_time()*1000)


model = MyModel()
modules = {
    "model": model
}
servicer = ModuleServiceServicer(table, modules)

servicer.run()