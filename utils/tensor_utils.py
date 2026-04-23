"""Tensor serialization/deserialization utilities using safetensors format.

Preserves ``requires_grad`` metadata through the safetensors ``__metadata__``
field so that gradient information survives the wire round-trip.
"""

import json
from typing import Dict

import safetensors.torch
import torch


def bool_to_str(value: bool) -> str:
    """Convert a boolean to its string representation for metadata storage."""
    return "True" if value else "False"


def str_to_bool(value: str) -> bool:
    """Convert a metadata string back to a boolean."""
    return value == "True"


def _get_metadata(data: bytes) -> Dict[str, str]:
    """Extract the ``__metadata__`` dictionary from a safetensors byte buffer.

    See https://github.com/huggingface/safetensors/issues/194#issuecomment-1466496698
    for the binary layout reference.
    """
    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    return header.get("__metadata__", {})


def serialize_tensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    """Serialize a dictionary of tensors to bytes using safetensors.

    The ``requires_grad`` flag of each tensor is preserved in the safetensors
    metadata so that :func:`deserialize_tensors` can restore it.
    """
    requires_grads: Dict[str, str] = {
        key: bool_to_str(tensors[key].requires_grad) for key in tensors
    }
    return safetensors.torch.save(tensors, metadata=requires_grads)


def deserialize_tensors(tensor_bytes: bytes) -> Dict[str, torch.Tensor]:
    """Deserialize tensors from a safetensors byte buffer.

    Restores the ``requires_grad`` flag from the metadata that was saved by
    :func:`serialize_tensors`.
    """
    tensors: Dict[str, torch.Tensor] = safetensors.torch.load(tensor_bytes)
    metadata: Dict[str, str] = _get_metadata(tensor_bytes)
    for key in tensors:
        tensors[key].requires_grad = str_to_bool(metadata[key]) if key in metadata else False
    return tensors


# Minimal placeholder bytes for error responses — a single zero-element tensor.
# Previously this used ``torch.Tensor(1337)`` which created an uninitialized
# tensor with 1337 elements (~10 KB of garbage data sent on every error).
DEFAULT_ZERO_SAFETENSOR_BYTES = serialize_tensors({"__placeholder__": torch.zeros(0)})