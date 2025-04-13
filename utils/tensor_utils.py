from numpy import save
import safetensors.torch
import torch
from typing import Dict
import json

def bool2str(x: bool) -> str:
    return "True" if x else "False"

def str2bool(x: str) -> bool:
    return x == "True"

def _get_metadata(data: bytes) -> Dict[str, str]: # https://github.com/huggingface/safetensors/issues/194#issuecomment-1466496698
    n_header: bytes = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes: bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    return header.get("__metadata__", {})
    

def serialize_tensors(tensors: Dict[str, torch.Tensor]) -> bytes:
    requires_grads: Dict[str, str] = {key: bool2str(tensors[key].requires_grad) for key in tensors.keys()}
    return safetensors.torch.save(tensors, metadata=requires_grads)

def deserialize_tensors(tensor_bytes: bytes) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = safetensors.torch.load(tensor_bytes)
    metadata: Dict[str, str] = _get_metadata(tensor_bytes)
    requires_grads: Dict[str, bool] = {key: str2bool(metadata[key]) for key in metadata.keys()}

    for key in tensors.keys():
        tensors[key].requires_grad = requires_grads[key]
    return tensors
     



DEFAULT_ZERO_SAFETENSOR_BYTES = serialize_tensors(
    {"deadbeef": torch.Tensor(1337)}
)