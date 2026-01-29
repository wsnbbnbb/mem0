"""
AMemo 主模块 - 模块化记忆系统
整合所有子模块并提供统一接口
"""

from typing import Any, Dict, List, Optional

from AgentMem.configs.base import MemoryConfig
from AgentMem.memory.base import MemoryBase

from .index import AMemoryIndex
from .add import MemoryAdder
from .search import MemorySearcher
from .response import MemoryResponder


class AMemo(MemoryBase):
    """
    AMemo - 模块化记忆系统
    
    整合以下四个核心功能：
    1. Add: 添加新记忆（文本向量 + 符号提取 + 图存储）
    2. Index: 索引和管理记忆（向量存储 + 历史记录）
    3. Search: 检索记忆（向量检索 + 符号过滤 + 重排序）
    4. Response: 生成响应（基于上下文的答案生成）
    """
    
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        """
        初始化 AMemo 系统
        
        Args:
            config: 系统配置
        """
        # 初始化索引模块（核心基础设施）
        self.index = AMemoryIndex(config)
        
        # 初始化添加模块
        self.adder = MemoryAdder(self.index)
        
        # 初始化检索模块
        self.searcher = MemorySearcher(self.index)
        
        # 初始化响应模块
        self.responder = MemoryResponder(self.index, self.searcher)
        
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'AMemo':
        """
        从配置字典创建 AMemo 实例
        
        Args:
            config_dict: 配置字典
            
        Returns:
            AMemo 实例
        """
        try:
            # 兼容旧配置
            if ("graph_store" in config_dict and 
                "vector_store" not in config_dict and 
                "embedder" in config_dict):
                config_dict["vector_store"] = {
                    "config": {
                        "embedding_model_dims": config_dict["embedder"]["config"]["embedding_dims"]
                    }
                }
            config = MemoryConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"配置验证失败: {e}")
        
        return cls(config)
    
    # ========== Add 方法 ==========
    
    def add(
        self, 
        messages: list, 
        user_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加对话消息到记忆系统
        
        向量存储 + 符号提取 + 图存储
        
        Args:
            messages: 消息列表，每个消息是字典格式 {"role": "user", "content": "..."}
            user_id: 用户ID
            metadata: 额外的元数据
            
        Returns:
            创建的内存ID
        """
        return self.adder.add_messages(messages, user_id, metadata)
    
    # ========== Index/Get 方法 ==========
    
    def get(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        获取特定记忆
        
        Args:
            user_id: 用户ID
            memory_id: 记忆ID
            
        Returns:
            记忆数据或None
        """
        return self.index.get_memory(user_id, memory_id)
    
    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的所有记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            记忆列表
        """
        return self.index.get_all_memories(user_id)
    
    def history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的记忆历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            历史记录列表
        """
        return self.index.get_history(user_id, limit)
    
    def update(
        self, 
        user_id: str, 
        memory_id: str, 
        new_text: str, 
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新记忆内容
        
        Args:
            user_id: 用户ID
            memory_id: 记忆ID
            new_text: 新的文本内容
            new_metadata: 新的元数据
            
        Returns:
            是否成功
        """
        return self.index.update_memory(user_id, memory_id, new_text, new_metadata)
    
    # ========== Search/Retrieve 方法 ==========
    
    def search(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        检索相关记忆
        
        向量检索 -> 符号过滤 -> 重排序
        
        Args:
            user_id: 用户ID
            query: 查询文本
            limit: 返回结果限制
            
        Returns:
            检索结果列表
        """
        return self.searcher.search(user_id, query, limit)
    
    # ========== Response 方法 ==========
    
    def ask(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        基于记忆回答问题
        
        Args:
            user_id: 用户ID
            query: 用户问题
            limit: 检索记忆数量限制
            include_context: 是否在响应中包含检索到的上下文
            
        Returns:
            包含响应和相关记忆的字典
        """
        return self.responder.generate_response(
            user_id, 
            query, 
            limit=limit,
            include_context=include_context
        )
    
    def summarize(
        self, 
        user_id: str, 
        topic: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        生成记忆摘要
        
        Args:
            user_id: 用户ID
            topic: 可选的主题过滤
            limit: 包含在摘要中的记忆数量
            
        Returns:
            包含摘要的字典
        """
        return self.responder.generate_summary(user_id, topic, limit)
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            统计信息字典
        """
        return self.responder.get_statistics(user_id)
    
    # ========== Delete 方法 ==========
    
    def delete(self, user_id: str, memory_id: str) -> bool:
        """
        删除特定记忆
        
        Args:
            user_id: 用户ID
            memory_id: 记忆ID
            
        Returns:
            是否成功
        """
        # 删除向量存储中的记忆
        self.index.vector_store.delete(ids=[memory_id])
        
        # 删除历史数据库中的记忆
        self.index.db.delete_memory(user_id, memory_id)
        
        # 删除图存储中的记忆节点（如果启用）
        if self.index.enable_graph and self.index.graph:
            self.index.graph.delete_memory_node(memory_id)
        
        return True
    
    def delete_all(
        self, 
        user_id: Optional[str] = None, 
        agent_id: Optional[str] = None, 
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        删除所有符合条件的记忆
        
        Args:
            user_id: 用户ID过滤
            agent_id: 代理ID过滤
            run_id: 运行ID过滤
            
        Returns:
            操作结果消息
        """
        return self.index.delete_all_memories(user_id, agent_id, run_id)
    
    # ========== 访问器方法 ==========
    
    def get_index(self) -> AMemoryIndex:
        """获取索引模块实例"""
        return self.index
    
    def get_adder(self) -> MemoryAdder:
        """获取添加模块实例"""
        return self.adder
    
    def get_searcher(self) -> MemorySearcher:
        """获取检索模块实例"""
        return self.searcher
    
    def get_responder(self) -> MemoryResponder:
        """获取响应模块实例"""
        return self.responder
    
    # ========== 兼容性属性 ==========
    
    @property
    def db(self):
        """历史数据库（兼容性）"""
        return self.index.db
    
    @property
    def embedding_model(self):
        """嵌入模型（兼容性）"""
        return self.index.embedding_model
    
    @property
    def vector_store(self):
        """向量存储（兼容性）"""
        return self.index.vector_store
    
    @property
    def llm(self):
        """LLM（兼容性）"""
        return self.index.llm
    
    @property
    def config(self):
        """配置（兼容性）"""
        return self.index.config
    
    @property
    def graph(self):
        """图存储（兼容性）"""
        return self.index.graph
    
    @property
    def enable_graph(self):
        """图存储启用状态（兼容性）"""
        return self.index.enable_graph
    
    @property
    def collection_name(self):
        """集合名称（兼容性）"""
        return self.index.collection_name


# ========== 兼容性类名 ==========

# 为向后兼容，保留 Memory 类名作为 AMemo 的别名
Memory = AMemo
