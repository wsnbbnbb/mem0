"""
AMemo 检索模块 - 负责双通道混合检索
[创新点 2: 检索 - 双通道混合检索与重排序]：向量检索 -> 符号过滤/重排序
"""

import json
import logging
from typing import Any, Dict, List, Optional

from AgentMem.logger import get_logger
from .index import AMemoryIndex
from .prompts import RE_RANKING_VALIDATION_PROMPT

logger = get_logger(__name__, filename="AMem.log")


class MemorySearcher:
    """
    内存检索器
    负责向量检索、符号过滤和重排序
    """
    
    def __init__(self, index: AMemoryIndex):
        """
        初始化内存检索器
        
        Args:
            index: 内存索引实例
        """
        self.index = index
        self.config = index.config
        self.llm = index.llm
        self.embedding_model = index.embedding_model
        self.vector_store = index.vector_store
        
    def search(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        双通道混合检索：向量检索 + 符号过滤/重排序
        
        Args:
            user_id: 用户ID
            query: 查询文本
            limit: 返回结果数量限制
            
        Returns:
            检索结果列表，按相关性排序
        """
        # 1. 语义通道：初次向量检索 (Vector Search)
        candidate_results = self._vector_search(user_id, query, limit * 3)
        
        if not candidate_results:
            logger.warning(
                f"No search results found for query: {query}, user_id: {user_id}"
            )
            return []
        
        # 2. 符号通道与推理：重排序与验证
        final_memories = self._re_rank_memories(query, candidate_results, limit)
        
        # 如果重排序失败或返回空，则回退到原始向量检索结果
        if not final_memories and candidate_results:
            logger.warning(
                "LLM re-ranking failed or returned empty, "
                "falling back to top vector results"
            )
            return [
                {**mem.copy(), "rank_reasoning": "Fallback (Vector Search)"}
                for mem in candidate_results[:limit]
            ]
            
        return final_memories

    def _vector_search(
        self, 
        user_id: str, 
        query: str, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        向量检索
        
        Args:
            user_id: 用户ID
            query: 查询文本
            limit: 候选结果限制
            
        Returns:
            候选结果列表
        """
        # 计算查询嵌入
        query_embedding = self.embedding_model.embed(query)
        
        # 向量检索
        candidate_results = self.vector_store.search(
            query=query, 
            vectors=query_embedding,
            limit=limit,
            filters={"user_id": user_id}
        )
        
        if not candidate_results:
            return []
        
        # 格式化候选记忆
        candidate_memories = []
        for result in candidate_results:
            processed = self._process_search_result(result)
            if processed:
                candidate_memories.append(processed)
        
        return candidate_memories

    def _process_search_result(
        self, 
        result: Any
    ) -> Optional[Dict[str, Any]]:
        """
        处理单个搜索结果
        
        Args:
            result: 原始搜索结果
            
        Returns:
            处理后的结果字典，如果无效则返回None
        """
        try:
            # 获取 payload（包含所有元数据）
            if hasattr(result, 'payload') and isinstance(result.payload, dict):
                payload = result.payload.copy()
            elif isinstance(result, dict):
                payload = result.copy()
            else:
                logger.debug(f"Skipping result: invalid type {type(result)}")
                return None
            
            # 获取 id
            mem_id = payload.pop('id', None)
            if mem_id is None:
                mem_id = getattr(result, 'id', None)
                if mem_id is None:
                    logger.debug(f"Skipping result: no id found")
                    return None
            
            # 获取 score
            score = payload.pop('score', None)
            if score is None:
                score = getattr(result, 'score', 0)
            
            # 获取 text (尝试多个可能的键)
            text = (
                payload.pop('data', None) or 
                payload.pop('text', None) or 
                payload.pop('memory', '')
            )
            
            # 跳过空的或无效的记忆
            if not text or not mem_id:
                logger.debug(
                    f"Skipping invalid result: id={mem_id}, "
                    f"text={text[:50] if text else 'empty'}"
                )
                return None
            
            return {
                "memory_id": mem_id,
                "id": mem_id,
                "text": text,
                "score": round(float(score), 4) if score is not None else 0.0,
                **payload
            }
        except Exception as e:
            logger.error(f"Error processing result: {e}, result type: {type(result)}")
            return None

    def _re_rank_memories(
        self, 
        query: str, 
        candidate_memories: List[Dict[str, Any]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        使用LLM对记忆进行重排序和验证
        
        Args:
            query: 原始查询
            candidate_memories: 候选记忆列表
            limit: 返回结果限制
            
        Returns:
            重排序后的记忆列表
        """
        # 构建重排序提示
        prompt = RE_RANKING_VALIDATION_PROMPT.format(
            query=query,
            candidate_memories=json.dumps(candidate_memories, indent=2)
        )
        
        try:
            # LLM 执行逻辑推理和重排序
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
            )
            
            # 解析响应
            if isinstance(response, str):
                response = eval(response)
            
            logger.info(
                f"LLM Re-ranking response: {response}, "
                f"type(response): {type(response)}, "
                f"filtered_memories: {response.get('filtered_memories', []) if isinstance(response, dict) else 'N/A'}"
            )
            
            re_ranked_list = response.get("filtered_memories", [])
            logger.info(f"Parsed re-ranked list: {re_ranked_list}")
            
            # 整合最终结果
            return self._merge_re_ranked_results(
                candidate_memories, 
                re_ranked_list, 
                limit
            )
                
        except Exception as e:
            logger.error(
                f"LLM Re-ranking failed with error: {e}. "
                f"Falling back to top vector results."
            )
            return []

    def _merge_re_ranked_results(
        self, 
        candidate_memories: List[Dict[str, Any]], 
        re_ranked_list: List[Dict[str, Any]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        合并重排序结果
        
        Args:
            candidate_memories: 原始候选记忆
            re_ranked_list: 重排序后的列表
            limit: 返回结果限制
            
        Returns:
            最终结果列表
        """
        final_memories = []
        logger.info(f"candidate_memories: {candidate_memories}")
        
        # 创建 ID 到原始记忆的映射
        id_to_memory = {mem['id']: mem for mem in candidate_memories}
        logger.info(f"id_to_memory: {id_to_memory}")
        
        # 按重排序顺序构建结果
        for item in re_ranked_list[:limit]:
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
        return final_memories
