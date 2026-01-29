import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytz
from typing import Any, Dict, Optional , List, Tuple
import logging
from pathlib import Path
from AgentMem.logger import get_logger
logger = get_logger(__name__, filename="AMem.log")

from pydantic import ValidationError

from AgentMem.configs.base import MemoryConfig
from AgentMem.memory.base import MemoryBase
from AgentMem.memory.setup import AgentMem_dir, setup_config
from AgentMem.memory.storage import SQLiteManager
from AgentMem.memory.telemetry import capture_event
from AgentMem.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory
from AgentMem.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
# 建议在 configs/prompts.py 中定义
SYMBOL_EXTRACTION_PROMPT = """
You are an intelligent knowledge extractor. Your task is to analyze the following memory chunk (a piece of user-agent conversation) and extract crucial structured information in JSON format.

Constraints:
1. Identify all main [Entities] (e.g., people, projects, places, dates).
2. Identify the [Core Relationship] or [Action] that links the entities (e.g., 'discusses', 'scheduled for', 'completed').
3. Extract the [Time Context] (exact date, day of the week, or relative term like 'next week'). If none, use 'N/A'.
4. Do not include any explanation or extra text. Output ONLY the JSON object.

Example Input: "User: Hey, did we finalize the Q3 marketing plan review? Agent: Yes, that was completed last Tuesday, November 15th, by Sarah and David."
Example Output: 
{{
  "Entities": ["Q3 marketing plan review", "Sarah", "David"],
  "Core Relationship": "completed",
  "Time Context": "November 15th"
}}

---
Memory Chunk: 
{memory_chunk}
"""

RE_RANKING_VALIDATION_PROMPT = """
You are a highly logical Re-ranker and Validator. A user asked the question: '{query}'.
The retrieval system provided the following candidate memory chunks (with their respective semantic relevance scores).

Candidate Memories:
{candidate_memories}

The initial vector search found these memories to be semantically relevant. However, you must now apply logical and symbolic constraints (based on entities, time, and relationships) to filter and re-rank them.

Instructions:
1. Filter out any memories that are factually contradicted by a high-ranking memory, or that are clearly irrelevant to the specific entities/time mentioned in the query.
2. Rank the remaining memories from 1 (Most Relevant) to N.

Output ONLY the final filtered and re-ranked memories in a JSON object with key "filtered_memories.If a memory must be discarded, exclude it.":
{{
  "filtered_memories": [
    {{ "rank": 1, "memory_id": "id-xyz", "reasoning": "Directly mentions the project status." }},
    {{ "rank": 2, "memory_id": "id-abc", "reasoning": "Provides background context." }}
  ]
}}
"""

class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        setup_config()
        self.config = config

        # embedding / vector store / llm
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)

        # sqlite 历史数据库
        self.db = SQLiteManager(self.config.history_db_path)

        # collection_name & path
        self.collection_name = self.config.vector_store.config.collection_name or "mem0migrations"
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            self.config.vector_store.config.path = os.path.join(AgentMem_dir, provider_path)
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

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            # 兼容旧配置
            if "graph_store" in config_dict and "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {
                    "config": {
                        "embedding_model_dims": config_dict["embedder"]["config"]["embedding_dims"]
                    }
                }
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(f"配置验证失败: {e}")
        return cls(config)
    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with data={data[:50] if isinstance(data, str) else data}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        payload = metadata or {}
        payload["data"] = data  # 用于检索时获取文本
        payload["text"] = data   # 兼容性字段
        payload["hash"] = hashlib.md5(data.encode()).hexdigest()
        payload["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
        
        logger.debug(f"Inserting memory {memory_id} with user_id={payload.get('user_id')}")

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[payload],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id
    def add(self, messages: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        [创新点 1: 写入 - 双重编码]：存储文本向量，并同时进行符号提取和图存储。
        Args:
            messages: 消息列表，每个消息是字典格式 {"role": "user", "content": "...", ...}
            user_id: 用户ID
            metadata: 额外的元数据
        """
        
        # # 1. 向量存储 (Vector Storage)
        # logger.info(f"输入文本:{text}")
        # for message in text:
        #     embedding = self.embedding_model.embed(message)
        # embedding = self.embedding_model.embed(text)
        # # logger.info(f"嵌入文本:{embedding}")
        # self.vector_store.insert(
        #     vectors=[embedding],
        #     ids=[memory_id],
        #     payloads=[{"id": memory_id, "user_id": user_id, "text": text, **(metadata or {})}],
        # )
        for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata) or {}
                per_msg_meta["user_id"] = user_id  # 确保 user_id 被包含
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)

        # 2. 符号提取与图存储 (Symbol Extraction and Graph Storage)
        if self.enable_graph and self.graph:
            try:
                # 使用 LLM 提取符号信息
                prompt = SYMBOL_EXTRACTION_PROMPT.format(memory_chunk=messages)
                
                # 假设 self.llm 有一个 generate_text 方法
                response = self.llm.generate_response(prompt)
                
                # 解析 LLM 的 JSON 输出
                symbolic_data = json.loads(response.strip())
                
                entities = symbolic_data.get("Entities", [])
                relationship = symbolic_data.get("Core Relationship", "mentions")
                time_context = symbolic_data.get("Time Context", None)

                # 将信息写入图数据库 (Graph Store)
                # 建立图节点和关系：(Entity A) -[RELATIONSHIP]-> (Entity B)
                if entities:
                    # 创建一个代表此记忆片段的中心节点
                    # self.graph.add_memory_node(memory_id, text, user_id, time_context) 
                    
                    # 连接实体到记忆节点
                    for entity in entities:
                        # 假设 graph.add_entity_link 方法可以创建实体节点和关系
                        self.graph._add_entities(entity, {"user_id":user_id,"agent_id":mem_id},relationship) 

                print(f"Added memory {mem_id} to vector store and extracted symbols.")
            except Exception as e:
                print(f"Warning: Failed to extract or store symbolic data for memory {mem_id}. Error: {e}")

        # 3. 历史数据库存储 (History DB Storage)
       
        return mem_id


    def search(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        [创新点 2: 检索 - 双通道混合检索与重排序]：向量检索 -> 符号过滤/重排序 -> 答案生成。
        """
        # 1. 语义通道：初次向量检索 (Vector Search)
        query_embedding = self.embedding_model.embed(query)
        
        candidate_results = self.vector_store.search(
            query = query, 
            vectors = query_embedding,
            limit = limit * 3,  # 提高召回限制，以便后续过滤
            filters={"user_id": user_id}
        )
        
        if not candidate_results:
            logger.warning(f"No search results found for query: {query}, user_id: {user_id}")
            return []
        
        # 格式化候选记忆，用于 LLM 重排序
        candidate_memories = []
        for result in candidate_results:
            # 处理不同类型的结果格式
            # 1. 如果是对象属性 (result.id, result.payload, result.score)
            # 2. 或者是字典格式的 payload
            # 3. payload 中可能用 'data' 键存储文本
            
            try:
                # 获取 payload（包含所有元数据）
                if hasattr(result, 'payload') and isinstance(result.payload, dict):
                    payload = result.payload.copy()  # 复制 payload 以避免修改原始数据
                elif isinstance(result, dict):
                    payload = result.copy()
                else:
                    logger.debug(f"Skipping result: invalid type {type(result)}")
                    continue
                
                # 获取 id
                mem_id = payload.pop('id', None)
                if mem_id is None:
                    mem_id = getattr(result, 'id', None)
                    if mem_id is None:
                        logger.debug(f"Skipping result: no id found")
                        continue
                
                # 获取 score
                score = payload.pop('score', None)
                if score is None:
                    score = getattr(result, 'score', 0)
                
                # 获取 text (尝试多个可能的键)
                text = payload.pop('data', None) or payload.pop('text', None) or payload.pop('memory', '')
                
                # 跳过空的或无效的记忆
                if not text or not mem_id:
                    logger.debug(f"Skipping invalid result: id={mem_id}, text={text[:50] if text else 'empty'}")
                    continue
                
                candidate_memories.append({
                    "memory_id": mem_id,  # 使用 memory_id 键名以兼容 LLM 重排序
                    "id": mem_id,  # 同时保留 id 键
                    "text": text,
                    "score": round(float(score), 4) if score is not None else 0.0,
                    # 保留所有其他元数据
                    **payload
                })
            except Exception as e:
                logger.error(f"Error processing result: {e}, result type: {type(result)}")
                continue
        
        # 2. 符号通道与推理：重排序与验证 (Re-ranking and Validation)
        
        # 2a. 提示词注入
        prompt = RE_RANKING_VALIDATION_PROMPT.format(
            query=query,
            candidate_memories=json.dumps(candidate_memories, indent=2)
        )
        
        # 2b. LLM 执行逻辑推理和重排序
        try:
            # response = self.llm.generate_response(prompt)
            response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
        )
            
            response = eval(response) if isinstance(response, str) else response
            logger.info(
    f"LLM Re-ranking response: {response}, "
    f"type(response): {type(response)}, "
    f"filtered_memories: {response.get('filtered_memories', []) if isinstance(response, dict) else 'N/A'}"
)
            # 假设 LLM 返回 JSON 格式的重排序结果
            # re_ranked_list: List[Dict[str, Any]] = json.loads(response["filtered_memories"].strip())
            re_ranked_list: List[Dict[str, Any]] = response.get("filtered_memories", [])
            logger.info(f"Parsed re-ranked list: {re_ranked_list}")
            # 3. 结果整合与最终输出 (Final Integration)
            final_memories = []
            logger.info(f"candidate_memories: {candidate_memories}")
            # 创建一个 ID 到原始记忆的映射
            id_to_memory = {mem['id']: mem for mem in candidate_memories}
            logger.info(f"id_to_memory: {id_to_memory}")
            for item in re_ranked_list[:limit]: # 限制最终输出数量
                mem_id = item.get('memory_id')
                logger.info(f"Processing re-ranked memory ID: {mem_id}")
                if mem_id and mem_id in id_to_memory:
                    # 查找原始记忆，包含所有元数据
                    original_memory = id_to_memory[mem_id]
                    # 保留所有原有字段，只添加 rank_reasoning
                    final_memory = original_memory.copy()
                    final_memory["rank_reasoning"] = item.get('reasoning')
                    final_memories.append(final_memory)
            logger.info(f"Final re-ranked memories: {final_memories}")
            # 如果 LLM 重排序失败或返回空，则回退到原始向量检索结果
            if not final_memories and candidate_memories:
                logger.warning("LLM re-ranking failed or returned empty, falling back to top vector results")
                return [mem.copy() for mem in candidate_memories[:limit]]
                
            return final_memories
            
        except Exception as e:
            logger.error(f"LLM Re-ranking failed with error: {e}. Falling back to top vector results.")
            # 失败回退机制 - 为 fallback 结果添加 rank_reasoning
            return [
                {**mem.copy(), "rank_reasoning": "Fallback (System Error)"}
                for mem in candidate_memories[:limit]
            ]
    # delete, get, get_all, history, update
    def delete(self, user_id: str, memory_id: str) -> bool:
        # 删除向量存储中的记忆
        self.vector_store.delete(ids=[memory_id])
        
        # 删除历史数据库中的记忆
        self.db.delete_memory(user_id, memory_id)
        
        # 删除图存储中的记忆节点（如果启用）
        if self.enable_graph and self.graph:
            self.graph.delete_memory_node(memory_id)
    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        memories = self.vector_store.list(filters=filters)[0]
        # for memory in memories:
        #     self._delete_memory(memory.id)
        for memory in memories:
            try:
                self._delete_memory(memory.id)
            except IndexError as e:
                logger.warning(f"Failed to delete memory {memory.id}: {e}")
        logger.info(f"Deleted {len(memories)} memories")

        return {"message": "Memories deleted successfully!"}
    def _delete_memory(self, memory_id: str):
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        # capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id
    def get(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        return self.db.get_memory(user_id, memory_id)
    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        return self.db.get_all_memories(user_id)
    def history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.db.get_recent_memories(user_id, limit)
    def update(self, user_id: str, memory_id: str, new_text: str, new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        # 更新向量存储中的记忆
        new_embedding = self.embedding_model.embed(new_text)
        self.vector_store.update(
            ids=[memory_id],
            vectors=[new_embedding],
            payloads=[{"user_id": user_id, "text": new_text, **(new_metadata or {})}]
        ) 
# --- 4. Main 测试逻辑 ---
def main():
    config = {
    "llm": {
         "provider": "vllm",
            "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
            "api_key": "vllm-api-key",
            "temperature": 0,
            "max_tokens": 2000,
            },
    },
        "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    # },
    #  "graph_store": {
    #         "provider": "neo4j",# or neo4j-community
    #         "config": {
    #             "url": "bolt://localhost:7687",
    #             "username": "neo4j", # or neo4j
    #             "password": "Neo4j2025",
    #             "database": "neo4j",
    #         }
    #     }
}
    print("--- 🧠 HRNSM 记忆系统对话文本测试 ---")
    memory = Memory.from_config(config)
    user_id = "test_user"

    # --- 测试 ADD (写入对话文本) ---
    print("\n## 📝 1. 测试 ADD (写入对话文本)")
    
    # 示例对话文本 1: 个人事件
    dialogue_1 = {"role":"user","content":"Melanie: Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday – it was really rewarding. Really made me think about taking care of our minds."}
    # memory.add(user_id, dialogue_1)
    
    # 示例对话文本 2: 关键事实
    dialogue_2 = {"role":"user","content": "Melanie: The Q4 Report review is scheduled for next Monday. Caroline: Perfect, I'll block out time for that."}
    
     # new_id = memory.add(user_id, dialogue_2)
    
    print(f"-> 已存储两个对话片段 (一个关于慈善跑，一个关于 Q4 报告)")
    messages = [dialogue_1, dialogue_2]
    memory.add(messages, user_id=user_id, metadata={"timestamp":"Jan28 2026"})
    # 验证符号提取是否被调用
    # print(f"-> 验证图存储调用: {memory.graph.add_memory_node.called}")

    print("\n---")
    
    # --- 测试 SEARCH (检索：逻辑过滤噪声) ---
    print("## 🔎 2. 测试 SEARCH (双通道混合检索与逻辑过滤)")
    # 查询：寻找一个明确的日程/事实
    query = "When is the Q4 Report review scheduled?"
    
    results = memory.search(user_id, query, limit=2)
    
    print(f"-> 搜索查询: '{query}'")
    print(f"-> **最终结果 (Top 2):**")
    
    if not results:
        print("搜索失败，返回空结果。")
        return

    for i, result in enumerate(results):
        print(f"\n- Rank {i+1}: (Score: {result['score']})")
        print(f"  Text: {result['text']}")
        print(f"  Logic: {result['rank_reasoning']}")
        print(f"  result:  {result}")
        
    # 预期分析：
    # 向量检索会返回 M2 (Q4) 和 M1 (慈善跑)
    # LLM 重排序会判断 M2 直接回答了问题，将其排在第一；M1 虽语义相关但不是事实答案，排在第二。
    # 噪音 M3 (Q1 报告) 被 LLM 逻辑过滤。

if __name__ == "__main__":
    main()