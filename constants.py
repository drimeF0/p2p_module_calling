"""Shared constants for the p2p_module_calling package.

``PUBLIC_INITIAL_PEERS`` can be set to a list of multi-address strings so that
different machines can discover each other through the Hivemind DHT.  Leave the
list empty for local testing — the DHT will bootstrap itself automatically.
"""

import os
from typing import List, Optional

# A list of Hivemind DHT multi-address strings for peer discovery.
# Override via the ``P2P_INITIAL_PEERS`` environment variable (comma-separated).
# Example: "/ip4/1.2.3.4/tcp/31337/p2p/QmFooBar"
PUBLIC_INITIAL_PEERS: Optional[List[str]] = (
    [addr.strip() for addr in os.environ.get("P2P_INITIAL_PEERS", "").split(",") if addr.strip()]
    or None
)