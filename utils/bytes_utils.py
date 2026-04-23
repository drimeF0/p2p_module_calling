"""Utilities for splitting large byte payloads into smaller chunks.

Used by the streaming RPC endpoints to avoid exceeding P2P message size limits.
"""

from typing import List


# Default chunk size: 10 KiB.  Chosen to stay well under typical P2P daemon
# message limits while keeping the number of round-trips manageable.
DEFAULT_CHUNK_SIZE = 1024 * 10


def split_bytes(byte_data: bytes, max_size: int = DEFAULT_CHUNK_SIZE) -> List[bytes]:
    """Split a bytes object into chunks of at most *max_size* bytes.

    Parameters
    ----------
    byte_data:
        The data to split.
    max_size:
        Maximum number of bytes per chunk (default 10 KiB).

    Returns
    -------
    List[bytes]
        A list of byte-string chunks whose concatenation equals *byte_data*.
    """
    return [byte_data[i : i + max_size] for i in range(0, len(byte_data), max_size)]
