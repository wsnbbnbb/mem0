import importlib.metadata
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"   # 默认版本号
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("mem0")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # fallback 默认版本


from AgentMem.client.main import AsyncMemoryClient, MemoryClient  # noqa
from AgentMem.memory.AMemo import Memory  # noqa
