from urllib import response
from grpclib.client import Channel
import torch
import safetensors.torch
from p2p_module_calling.proto import module_service 

import asyncio

async def run():
    channel = Channel('localhost', 50051)
    stub = module_service.ModuleServiceStub(channel)
    response = await stub.register_module(module_service.ModuleRegistrationRequest(module_id='my_module', module_bytes=b'my_module_bytes'))
    print(f"Module registered: {response.success}")
if __name__ == '__main__':
    asyncio.run(run())
