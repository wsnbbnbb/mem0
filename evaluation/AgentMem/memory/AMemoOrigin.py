# mem0/memory/main.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from copy import deepcopy
from typing import Any, Dict, Optional , List, Tuple
import uuid
import json
import logging
from pathlib import Path
# from ..logger import get_logger
# # ---------- 日志配置 ----------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)  # 如果 logs 文件夹不存在就创建

LOG_FILE = LOG_DIR / "AMemo.log"

# # 创建 logger
log = logging.getLogger("AMemoLogger")
log.setLevel(logging.INFO)  # 输出级别

# 控制台 handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

# 文件 handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)

# 添加 handler
# log.addHandler(console_handler)
log.addHandler(file_handler)

from pydantic import ValidationError

from AgentMem.configs.base import MemoryConfig
from AgentMem.memory.base import MemoryBase
from AgentMem.memory.setup import mem0_dir, setup_config
from AgentMem.memory.storage import SQLiteManager
from AgentMem.memory.telemetry import capture_event
from AgentMem.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory

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

Output ONLY the final, filtered, and re-ranked list of memories in the following JSON format. If a memory must be discarded, exclude it.

Example Output:
[
  {{ "rank": 1, "memory_id": "id-xyz", "reasoning": "Directly mentions the project status and time requested." }},
  {{ "rank": 2, "memory_id": "id-abc", "reasoning": "Provides background context about the project's inception." }}
]
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
            self.config.vector_store.config.path = os.path.join(mem0_dir, provider_path)
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
    
    def add(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        [创新点 1: 写入 - 双重编码]：存储文本向量，并同时进行符号提取和图存储。
        """
        memory_id = str(uuid.uuid4())
        
        # 1. 向量存储 (Vector Storage)
        embedding = self.embedding_model.embed(text)
        self.vector_store.insert(
            vectors=[embedding],
            payloads=[{"id": memory_id, "user_id": user_id, "text": text, **(metadata or {})}],
            ids=[memory_id]
        )

        # 2. 符号提取与图存储 (Symbol Extraction and Graph Storage)
        if self.enable_graph and self.graph:
            try:
                # 使用 LLM 提取符号信息
                prompt = SYMBOL_EXTRACTION_PROMPT.format(memory_chunk=text)
                
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
                        self.graph._add_entities(entity, {"user_id":user_id,"agent_id":memory_id},relationship) 

                print(f"Added memory {memory_id} to vector store and extracted symbols.")
            except Exception as e:
                print(f"Warning: Failed to extract or store symbolic data for memory {memory_id}. Error: {e}")

        # 3. 历史数据库存储 (History DB Storage)
        self.db.add_history(user_id, memory_id, text, metadata)

        capture_event("AgentMem.add", self, {"user_id": user_id, "text_len": len(text)})
        return memory_id


    def search(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        [创新点 2: 检索 - 双通道混合检索与重排序]：向量检索 -> 符号过滤/重排序 -> 答案生成。
        """
        # 1. 语义通道：初次向量检索 (Vector Search)
        query_embedding = self.embedding_model.embed(query)
        
        # 假设 vector_store.search 返回 (text, metadata, score)
        candidate_results: List[Tuple[str, Dict[str, Any], float]] = self.vector_store.search(
            query = query, 
            vectors = query_embedding,
            limit = limit * 3,  # 提高召回限制，以便后续过滤
            filters={"user_id": user_id}
        )
        if not candidate_results:
            return []
        # print(f"{candidate_results[0]}\n--------------------")
        # 格式化候选记忆，用于 LLM 重排序
        candidate_memories = []
        for result in candidate_results:
            id = result.id
            text = result.payload.get("text", "")
            score = result.score 
            candidate_memories.append({
                "memory_id": id,
                "text": text,
                "score": round(score, 4) 
            })
        
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
            log.info(f"LLM Re-ranking response: {response}")
            # 假设 LLM 返回 JSON 格式的重排序结果
            re_ranked_list: List[Dict[str, Any]] = json.loads(response.strip())
    
            # 3. 结果整合与最终输出 (Final Integration)
            final_memories = []
            log.info(f"candidate_memories: {candidate_memories}")
            # 创建一个 ID 到原始记忆的映射
            id_to_memory = {mem['memory_id']: mem for mem in candidate_memories}
            log.info(f"id_to_memory: {id_to_memory}")
            for item in re_ranked_list[:limit]: # 限制最终输出数量
                mem_id = item.get('memory_id')
                log.info(f"Processing re-ranked memory ID: {mem_id}")
                if mem_id and mem_id in id_to_memory:
                    # 查找原始记忆文本和分数
                    original_memory = id_to_memory[mem_id]
                    final_memories.append({
                        "id": mem_id,
                        "text": original_memory['text'],
                        "score": original_memory['score'],
                        "rank_reasoning": item.get('reasoning') # 包含重排序的逻辑解释
                    })
            log.info(f"Final re-ranked memories: {final_memories}")
            # 如果 LLM 重排序失败或返回空，则回退到原始向量检索结果
            if not final_memories and candidate_results:
                print("Warning: LLM re-ranking failed, falling back to top vector results.")
                return [{
                    "id": (getattr(res, 'id', None) or (res.payload.get('id') if isinstance(res.payload, dict) else None)),
                    "text": (res.payload.get('text') if isinstance(res.payload, dict) else getattr(res, 'text', '')),
                    "score": getattr(res, 'score', None),
                    "rank_reasoning": "Fallback (LLM re-ranking failure)"
                } for res in candidate_results[:limit]]
                
            return final_memories
            
        except Exception as e:
            print(f"Warning: LLM Re-ranking failed with error: {e}. Falling back to top vector results.")
            # 失败回退机制
            return [{
                "id": (getattr(res, 'id', None) or (res.payload.get('id') if isinstance(res.payload, dict) else None)),
                "text": (res.payload.get('text') if isinstance(res.payload, dict) else getattr(res, 'text', '')),
                "score": getattr(res, 'score', None),
                "rank_reasoning": "Fallback (System Error)"
            } for res in candidate_results[:limit]]
    # delete, get, get_all, history, update
    def delete(self, user_id: str, memory_id: str) -> bool:
        # 删除向量存储中的记忆
        self.vector_store.delete(ids=[memory_id])
        
        # 删除历史数据库中的记忆
        self.db.delete_memory(user_id, memory_id)
        
        # 删除图存储中的记忆节点（如果启用）
        if self.enable_graph and self.graph:
            self.graph.delete_memory_node(memory_id)
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
        "provider": "ollama",
        "config": {
            # "model": "qwen3:8b",
            "model": "gpt-oss:20b",
            "temperature": 0.1,
            "max_tokens": 4096,
        }
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
    dialogue_1 = "Melanie: Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday – it was really rewarding. Really made me think about taking care of our minds."
    memory.add(user_id, dialogue_1)
    
    # 示例对话文本 2: 关键事实
    dialogue_2 = "Melanie: The Q4 Report review is scheduled for next Monday. Caroline: Perfect, I'll block out time for that."
    new_id = memory.add(user_id, dialogue_2)
    
    print(f"-> 已存储两个对话片段 (一个关于慈善跑，一个关于 Q4 报告)")
    
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
        
    # 预期分析：
    # 向量检索会返回 M2 (Q4) 和 M1 (慈善跑)
    # LLM 重排序会判断 M2 直接回答了问题，将其排在第一；M1 虽语义相关但不是事实答案，排在第二。
    # 噪音 M3 (Q1 报告) 被 LLM 逻辑过滤。

if __name__ == "__main__":
    main()