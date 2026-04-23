"""Client-side module for invoking remote PyTorch modules over P2P.

Provides :class:`Client` for raw RPC calls and :class:`RemoteModule` which
wraps the remote call as a standard ``torch.nn.Module`` with full autograd
support — calling ``.backward()`` on the output automatically triggers a
remote backward pass.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

from hivemind import DHT
from hivemind.p2p import P2P, PeerID, StubBase
from hivemind.utils.logging import get_logger

from p2p_module_calling.module_service import (
    ModuleBackwardRequest,
    ModuleBackwardResponse,
    ModuleForwardRequest,
    ModuleForwardResponse,
)
from p2p_module_calling.server.module_servicer import ModuleServicer
from p2p_module_calling.utils import (
    deserialize_tensors,
    fill_dict,
    flatten,
    serialize_tensors,
    unfill_dict,
)
from p2p_module_calling.client.remote_module_worker import RemoteModuleWorker

logger = get_logger(__name__)

# A zero-size tensor used as a dummy first argument to ``_RemoteModuleCall.apply``
# so that autograd treats it as a differentiable output.  The tensor itself
# carries no data; it only exists to anchor the autograd graph.
_DUMMY = torch.empty(0, requires_grad=True)


class RemoteModuleError(Exception):
    """Raised when a remote module RPC returns an error response."""


def _get_server_stub(p2p: P2P, server_peer_id: PeerID) -> StubBase:
    """Create a P2P RPC stub for communicating with a remote ``ModuleServicer``."""
    return ModuleServicer.get_stub(p2p, server_peer_id)


class Client:
    """Low-level client for sending forward/backward RPCs to a remote module server.

    Parameters
    ----------
    dht:
        A running Hivemind DHT instance used to discover and connect to the peer.
    server_peer_id:
        The ``PeerID`` of the remote server hosting the modules.
    """

    def __init__(self, dht: DHT, server_peer_id: PeerID):
        # Replicate the P2P transport on a dedicated event-loop thread so that
        # synchronous callers are not blocked by the DHT's own loop.
        self._p2p: P2P = RemoteModuleWorker.run_coroutine(dht.replicate_p2p())
        self._stub = _get_server_stub(self._p2p, server_peer_id)

    # -- public API -----------------------------------------------------------

    def forward(self, module_id: str, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run a forward pass on the remote module and return the output tensors.

        Parameters
        ----------
        module_id:
            Identifier of the module registered on the server.
        data:
            Keyword-argument tensors to pass to the module's ``forward()``.

        Raises
        ------
        RemoteModuleError
            If the server reports that the call failed (e.g. module not found).
        """
        data_bytes = serialize_tensors(data)
        message = ModuleForwardRequest(module_id=module_id, input_tensor_bytes=data_bytes)
        result: ModuleForwardResponse = RemoteModuleWorker.run_coroutine(
            self._stub.rpc_forward_module(message),
        )
        if not result.success:
            raise RemoteModuleError(
                result.error_message or "Forward pass failed with no error message",
            )
        return deserialize_tensors(result.output_tensor_bytes)

    def backward(
        self,
        module_id: str,
        data: Dict[str, torch.Tensor],
        grad: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run a backward pass on the remote module and return input gradients.

        Parameters
        ----------
        module_id:
            Identifier of the module registered on the server.
        data:
            The original input tensors (will be re-forwarded on the server to
            rebuild the computation graph under ``torch.enable_grad()``).
        grad:
            Gradient tensors with respect to the module's outputs.

        Raises
        ------
        RemoteModuleError
            If the server reports that the call failed.
        """
        data_bytes = serialize_tensors(data)
        grad_bytes = serialize_tensors(grad)
        message = ModuleBackwardRequest(
            module_id=module_id,
            input_tensor_bytes=data_bytes,
            grad_tensor_bytes=grad_bytes,
        )
        result: ModuleBackwardResponse = RemoteModuleWorker.run_coroutine(
            self._stub.rpc_backward_module(message),
        )
        if not result.success:
            raise RemoteModuleError(
                result.error_message or "Backward pass failed with no error message",
            )
        return deserialize_tensors(result.grad_tensor_bytes)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Shut down the P2P connection and release resources."""
        if self._p2p is not None:
            try:
                RemoteModuleWorker.run_coroutine(self._p2p.shutdown())
            except Exception:
                logger.warning("Failed to shut down P2P connection cleanly", exc_info=True)
            self._p2p = None

    def __del__(self) -> None:
        # Best-effort cleanup — avoid raising during interpreter shutdown.
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class RemoteModule(torch.nn.Module):
    """A ``torch.nn.Module`` that forwards its computation to a remote server.

    The remote call is autograd-friendly: calling ``.backward()`` on the
    output triggers a remote backward pass, and gradients are propagated back
    to the local tensors exactly as if the computation were local.

    Parameters
    ----------
    client:
        An initialised :class:`Client` connected to the remote server.
    module_id:
        The identifier of the module on the server.
    """

    def __init__(self, client: Client, module_id: str):
        super().__init__()
        self.client = client
        self.module_id = module_id

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tensors, keys = flatten(x)
        return _RemoteModuleCall.apply(_DUMMY, self.module_id, self.client, keys, *tensors)


class _RemoteModuleCall(torch.autograd.Function):
    """Internal autograd function that performs a remote forward/backward pair.

    During the forward pass the inputs are serialized and sent to the server.
    During the backward pass the saved inputs and output gradients are sent
    back so that the server can re-run the forward pass under
    ``torch.enable_grad()`` and then call ``torch.autograd.backward()`` to
    obtain input gradients.

    Reference: https://github.com/learning-at-home/hivemind/blob/master/hivemind/moe/client/expert.py#L194
    """

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        uid: str,
        client: Client,
        inputs_keys: List[str],
        *inputs_tensors: torch.Tensor,
    ) -> torch.Tensor:
        inputs = {
            key: tensor.cpu().detach() for key, tensor in zip(inputs_keys, inputs_tensors)
        }
        ctx.uid = uid
        ctx.client = client
        ctx.keys = inputs_keys
        ctx.save_for_backward(*inputs_tensors)
        result = client.forward(uid, inputs)["result"]
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        grad_outputs_cpu = [tensor.cpu().contiguous() for tensor in grad_outputs]
        restored_inputs = dict(zip(ctx.keys, ctx.saved_tensors))
        grad_inputs = fill_dict(grad_outputs_cpu)
        result = ctx.client.backward(ctx.uid, restored_inputs, grad_inputs)
        flat_grads = unfill_dict(result)
        logger.debug("Backward pass completed for module %s", ctx.uid)
        return (_DUMMY, None, None, None, *flat_grads)