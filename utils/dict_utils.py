"""Utilities for converting between tensor dictionaries and flat lists.

These helpers exist because :class:`torch.autograd.Function` operates on
positional ``*args`` (flat tensors) while the public API exposes named
tensors via ``Dict[str, torch.Tensor]``.
"""

import torch
from typing import Dict, List, Tuple


def unpack_dict(
    tensors: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], List[str]]:
    """Split a tensor dictionary into a parallel list of tensors and list of keys.

    Example::

        >>> unpack_dict({"x": t1, "y": t2})
        ([t1, t2], ["x", "y"])
    """
    keys = list(tensors.keys())
    values = [tensors[key] for key in keys]
    return values, keys


def pack_to_dict(values: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pack a list of tensors into a dictionary with integer-string keys.

    This is the inverse of :func:`values_from_dict` for the simple case
    where keys are sequential indices.

    Example::

        >>> pack_to_dict([t1, t2])
        {"0": t1, "1": t2}
    """
    return {str(i): values[i] for i in range(len(values))}


def values_from_dict(values: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Extract tensors from a dictionary, preserving insertion order.

    Example::

        >>> values_from_dict({"0": t1, "1": t2})
        [t1, t2]
    """
    return list(values.values())


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated names from the original codebase)
# ---------------------------------------------------------------------------
flatten = unpack_dict          # old name: confusing — doesn't "flatten" nesting
fill_dict = pack_to_dict       # old name: unclear direction
unfill_dict = values_from_dict # old name: unclear direction