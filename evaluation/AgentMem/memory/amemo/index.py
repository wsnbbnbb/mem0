"""
AMemo 索引模块 - 负责内存创建、向量存储和索引管理
[创新点 1: 写入 - 双重编码]：存储文本向量
"""

import hashlib
import os
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz

from AgentMem.configs.base import MemoryConfig
from AgentMem.logger import get_logger
from AgentMem.memory.setup import AgentMem_dir, setup_config
from AgentMem.memory.storage import SQLiteManager
from AgentMem.memory.telemetry import capture_event
from AgentMem.utils.factory import EmbedderFactory, VectorStoreFactory, LlmFactory

logger = get_logger(__name__, filename="AMem.log")


class AMemoryIndex:
    """
    内存索引管理器
    负责向量存储、内存创建和历史记录
    """
    
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        setup_config()
        self.config = config

        # 初始化组件
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, 
            self.config.vector_store.config
        )
        self.llm = LlmFactory.create(
            self.config.llm.provider, 
            self.config.llm.config
        )

        # SQLite 历史数据库
        self.db = SQLiteManager(self.config.history_db_path)

        # import os here to avoid circular dependencies
        import os
        
        # collection_name & path
        self.collection_name = self.config.vector_store.config.collection_name or "mem0migrations"
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            self.config.vector_store.config.path = os.path.join(
                AgentMem_dir, 
                provider_path
            )
            os.makedirs(self.config.vector_store.config.path, exist_ok=True)

        # 图存储（可选）
        self.enable_graph = False
        self.graph = None
        if self.config.graph_store.config:
            if self.config.graph_store.provider == "memgraph":
                from AgentMem.memory.memgraph_memory import MemoryGraph
            elif self.config.graph_store.provider == "neptune":
                from AgentMem.graphs.neptune.main import MemoryGraph
            else:
                from AgentMem.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

        # telemetry
        capture_event("AgentMem.init", self, {"sync_type": "sync"})

    def create_memory(
        self, 
        data: str, 
        existing_embeddings: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建并索引单个内存
        
        Args:
            data: 要存储的文本内容
            existing_embeddings: 预计算的嵌入向量（可选）
            metadata: 附加元数据
            
        Returns:
            内存ID
        """
        logger.debug(
            f"Creating memory with data={data[:50] if isinstance(data, str) else data}"
        )
        
        # 计算嵌入向量
        if existing_embeddings and data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        
        # 生成唯一ID
        memory_id = str(uuid.uuid4())
        
        # 准备payload
        payload = metadata or {}
        payload["data"] = data  # 用于检索时获取文本
        payload["text"] = data   # 兼容性字段
        payload["hash"] = hashlib.md5(data.encode()).hexdigest()
        payload["created_at"] = datetime.now(
            pytz.timezone("US/Pacific")
        ).isoformat()
        
        logger.debug(
            f"Inserting memory {memory_id} with user_id={payload.get('user_id')}"
        )

        # 插入向量存储
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[payload],
        )
        
        # 添加到历史记录
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        
        capture_event(
            "mem0._create_memory", 
            self, 
            {"memory_id": memory_id, "sync_type": "sync"}
        )
        
        return memory_id

    def get_memory(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定内存
        
        Args:
            user_id: 用户ID
            memory_id: 内存ID
            
        Returns:
            内存数据或None
        """
        return self.db.get_memory(user_id, memory_id)

    def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有内存
        
        Args:
            user_id: 用户ID
            
        Returns:
            内存列表
        """
        return self.db.get_all_memories(user_id)

    def get_history(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        获取最近的内存历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            历史记录列表
        """
        return self.db.get_recent_memories(user_id, limit)

    def update_memory(
        self, 
        user_id: str, 
        memory_id: str, 
        new_text: str, 
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新内存内容
        
        Args:
            user_id: 用户ID
            memory_id: 内存ID
            new_text: 新的文本内容
            new_metadata: 新的元数据
            
        Returns:
            是否成功
        """
        # 更新向量存储
        new_embedding = self.embedding_model.embed(new_text)
        self.vector_store.update(
            ids=[memory_id],
            vectors=[new_embedding],
            payloads=[{"user_id": user_id, "text": new_text, **(new_metadata or {})}]
        )
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除单个内存
        
        Args:
            memory_id: 内存ID
            
        Returns:
            是否成功
        """
        # 获取现有内存
        existing_memory = self.vector_store.get(vector_id=memory_id)
        if not existing_memory:
            return False
            
        prev_value = existing_memory.payload["data"]
        
        # 删除向量存储
        self.vector_store.delete(vector_id=memory_id)
        
        # 添加删除历史
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        
        return True

    def delete_all_memories(
        self, 
        user_id: Optional[str] = None, 
        agent_id: Optional[str] = None, 
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        删除所有符合条件的内存
        
        Args:
            user_id: 用户ID过滤
            agent_id: 代理ID过滤
            run_id: 运行ID过滤
            
        Returns:
            操作结果消息
        """
        from AgentMem.memory.utils import process_telemetry_filters
        
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. "
                "If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.delete_all", 
            self, 
            {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )
        
        memories = self.vector_store.list(filters=filters)[0]
        
        for memory in memories:
            try:
                self.delete_memory(memory.id)
            except IndexError as e:
                logger.warning(f"Failed to delete memory {memory.id}: {e}")
        
        logger.info(f"Deleted {len(memories)} memories")
        return {"message": f"Deleted {len(memories)} memories successfully!"}
