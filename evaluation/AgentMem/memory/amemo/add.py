"""
AMemo 添加模块 - 负责符号提取和图存储
[创新点 1: 写入 - 双重编码]：符号提取和图存储
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from AgentMem.logger import get_logger
from .index import AMemoryIndex
from .prompts import SYMBOL_EXTRACTION_PROMPT

logger = get_logger(__name__, filename="AMem.log")


class MemoryAdder:
    """
    内存添加器
    负责解析消息、创建内存、提取符号并存储到图数据库
    """
    
    def __init__(self, index: AMemoryIndex):
        """
        初始化内存添加器
        
        Args:
            index: 内存索引实例
        """
        self.index = index
        self.config = index.config
        self.llm = index.llm
        self.enable_graph = index.enable_graph
        self.graph = index.graph
        
    def add_messages(
        self, 
        messages: List[Dict[str, Any]], 
        user_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加消息列表到记忆系统
        
        Args:
            messages: 消息列表，每个消息是字典格式 {"role": "user", "content": "...", ...}
            user_id: 用户ID
            metadata: 额外的元数据
            
        Returns:
            最后创建的内存ID
        """
        last_memory_id = None
        
        # 处理每条消息
        for message_dict in messages:
            if (
                not isinstance(message_dict, dict)
                or message_dict.get("role") is None
                or message_dict.get("content") is None
            ):
                logger.warning(f"Skipping invalid message format: {message_dict}")
                continue

            # 跳过系统消息
            if message_dict["role"] == "system":
                continue

            # 准备元数据
            per_msg_meta = deepcopy(metadata) or {}
            per_msg_meta["user_id"] = user_id  # 确保 user_id 被包含
            per_msg_meta["role"] = message_dict["role"]

            actor_name = message_dict.get("name")
            if actor_name:
                per_msg_meta["actor_id"] = actor_name

            # 创建内存
            msg_content = message_dict["content"]
            mem_id = self.index.create_memory(
                msg_content, 
                None,  # existing_embeddings
                per_msg_meta
            )
            last_memory_id = mem_id

        # 符号提取与图存储
        if last_memory_id and self.enable_graph and self.graph:
            self._extract_and_store_symbols(messages, user_id, last_memory_id)

        return last_memory_id

    def _extract_and_store_symbols(
        self, 
        messages: List[Dict[str, Any]], 
        user_id: str, 
        memory_id: str
    ) -> None:
        """
        提取符号信息并存储到图数据库
        
        Args:
            messages: 消息列表
            user_id: 用户ID
            memory_id: 内存ID
        """
        try:
            # 使用 LLM 提取符号信息
            prompt = SYMBOL_EXTRACTION_PROMPT.format(memory_chunk=messages)
            
            # 调用 LLM 生成响应
            response = self.llm.generate_response(prompt)
            
            # 解析 LLM 的 JSON 输出
            try:
                # 尝试解析 JSON
                if isinstance(response, str):
                    response = response.strip()
                    # 移除可能存在的代码块标记
                    if response.startswith("```"):
                        response = response.split("```")[1]
                        if response.startswith("json"):
                            response = response[4:]
                    symbolic_data = json.loads(response.strip())
                else:
                    symbolic_data = response
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}, response: {response}")
                return

            # 提取符号数据
            entities = symbolic_data.get("Entities", [])
            relationship = symbolic_data.get("Core Relationship", "mentions")
            time_context = symbolic_data.get("Time Context", None)

            # 将信息写入图数据库
            # 建立图节点和关系：(Entity A) -[RELATIONSHIP]-> (Entity B)
            if entities:
                # 连接实体到记忆节点
                for entity in entities:
                    # 假设 graph._add_entities 方法可以创建实体节点和关系
                    self.graph._add_entities(
                        entity, 
                        {"user_id": user_id, "agent_id": memory_id},
                        relationship
                    )

                logger.info(
                    f"Added memory {memory_id} to vector store and "
                    f"extracted symbols: {entities}"
                )
                
        except Exception as e:
            logger.warning(
                f"Warning: Failed to extract or store symbolic data for "
                f"memory {memory_id}. Error: {e}"
            )

    def add(
        self, 
        text: str, 
        user_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加单个文本到记忆系统（兼容方法）
        
        Args:
            text: 要添加的文本
            user_id: 用户ID
            metadata: 额外的元数据
            
        Returns:
            内存ID
        """
        # 将文本包装为消息格式
        messages = [{"role": "user", "content": text}]
        return self.add_messages(messages, user_id, metadata)
