import torch
from typing import Dict, List, Tuple



def flatten(tensors: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[str]]:
    flatten_tensors: List[torch.Tensor] = []
    keys: List[str] = []
    for key in tensors.keys():
        flatten_tensors.append(tensors[key])
        keys.append(key)
    return flatten_tensors, keys

def fill_dict(values: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {str(i): values[i] for i in range(len(values))}

def unfill_dict(values: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    return list(values.values())