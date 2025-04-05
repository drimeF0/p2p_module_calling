from numpy import save
import safetensors.torch
import torch
from typing import Dict


def deserialize_tensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    return safetensors.torch.save(tensors)

def serialize_tensors(tensor_bytes: bytes) -> Dict[str, torch.Tensor]:
    return safetensors.torch.load(tensor_bytes)


DEFAULT_ZERO_SAFETENSOR_BYTES = deserialize_tensors(
    {"deadbeef": torch.Tensor(1337)}
)