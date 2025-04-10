# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: module_service_proto.proto
# plugin: python-betterproto
# This file has been @generated

from dataclasses import dataclass
from typing import Optional

import betterproto


@dataclass(eq=False, repr=False)
class ModuleForwardRequest(betterproto.Message):
    module_id: str = betterproto.string_field(1)
    input_tensor_bytes: bytes = betterproto.bytes_field(2)


@dataclass(eq=False, repr=False)
class ModuleForwardResponse(betterproto.Message):
    success: bool = betterproto.bool_field(1)
    output_tensor_bytes: bytes = betterproto.bytes_field(2)
    error_message: Optional[str] = betterproto.string_field(3, optional=True)


@dataclass(eq=False, repr=False)
class ModuleBackwardRequest(betterproto.Message):
    module_id: str = betterproto.string_field(1)
    input_tensor_bytes: Optional[bytes] = betterproto.bytes_field(2, optional=True)
    grad_tensor_bytes: Optional[bytes] = betterproto.bytes_field(3, optional=True)


@dataclass(eq=False, repr=False)
class ModuleBackwardResponse(betterproto.Message):
    success: bool = betterproto.bool_field(1)
    grad_tensor_bytes: bytes = betterproto.bytes_field(2)
    error_message: Optional[str] = betterproto.string_field(3, optional=True)
