import torch
from torch.autograd import Function
from typing import Dict, Tuple, Optional
from torch.autograd.function import once_differentiable

from p2p_module_calling.client import Client, RemoteModule