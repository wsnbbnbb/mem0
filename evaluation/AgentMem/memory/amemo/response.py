"""
AMemo 响应生成模块 - 负责基于检索结果生成响应
[创新点 3: 响应 - 上下文感知答案生成]
"""

import logging
from typing import Any, Dict, List, Optional

from AgentMem.logger import get_logger
from .index import AMemoryIndex
from .search import MemorySearcher

logger = get_logger(__name__, filename="AMem.log")


class MemoryResponder:
    """
    内存响应器
    负责基于检索到的记忆生成上下文感知的响应
    """
    
    def __init__(self, index: AMemoryIndex, searcher: MemorySearcher):
        """
        初始化内存响应器
        
        Args:
            index: 内存索引实例
            searcher: 内存检索器实例
        """
        self.index = index
        self.searcher = searcher
        self.llm = index.llm
        
    def generate_response(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 5,
        include_context: bool = True,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        基于检索的记忆生成响应
        
        Args:
            user_id: 用户ID
            query: 用户查询
            limit: 检索记忆的数量限制
            include_context: 是否在响应中包含检索到的上下文
            response_format: 响应格式选项
            
        Returns:
            包含响应和上下文的字典
        """
        # 1. 检索相关记忆
        memories = self.searcher.search(user_id, query, limit=limit)
        
        if not memories:
            logger.warning(f"No memories found for query: {query}")
            return {
                "query": query,
                "response": "I don't have any relevant memories to help with that query.",
                "memories": [],
                "context": "",
                "success": False
            }
        
        # 2. 构建上下文
        context = self._build_context(memories)
        
        # 3. 生成响应
        if include_context:
            response = self._generate_response_with_context(query, context, response_format)
        else:
            response = self._generate_response_without_context(query, memories, response_format)
        
        # 4. 返回结果
        return {
            "query": query,
            "response": response,
            "memories": memories,
            "context": context,
            "success": True
        }
    
    def _build_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        基于检索到的记忆构建上下文字符串
        
        Args:
            memories: 检索到的记忆列表
            
        Returns:
            上下文字符串
        """
        if not memories:
            return ""
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            reasoning = memory.get('rank_reasoning', '')
            text = memory.get('text', '')
            
            part = f"Memory {i}:\n"
            part += f"  Content: {text}\n"
            if reasoning:
                part += f"  Relevance: {reasoning}\n"
            
            context_parts.append(part)
        
        return "\n".join(context_parts)
    
    def _generate_response_with_context(
        self, 
        query: str, 
        context: str, 
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        基于上下文生成响应
        
        Args:
            query: 用户查询
            context: 上下文字符串
            response_format: 响应格式选项
            
        Returns:
            生成的响应
        """
        prompt = f"""Based on the following memories, please answer the user's question.

User Question: {query}

Relevant Memories:
{context}

Instructions:
1. Use the information from the memories to provide an accurate and helpful response.
2. If the memories don't contain enough information to answer the question, acknowledge this limitation.
3. Cite specific memories when relevant.
4. Be concise and direct.

Response:"""

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers based on provided memories."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format
            )
            return response
        except Exception as e:
            logger.error(f"Failed to generate response with context: {e}")
            return "I apologize, but I encountered an error generating a response based on the retrieved memories."
    
    def _generate_response_without_context(
        self, 
        query: str, 
        memories: List[Dict[str, Any]],
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        不使用完整上下文生成响应（简化版本）
        
        Args:
            query: 用户查询
            memories: 检索到的记忆列表
            response_format: 响应格式选项
            
        Returns:
            生成的响应
        """
        # 提取最相关的记忆内容
        if memories:
            top_memory = memories[0].get('text', '')
            return f"Based on my memories: {top_memory}"
        else:
            return "I don't have any relevant memories for this query."
    
    def generate_summary(
        self, 
        user_id: str, 
        topic: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        生成用户的记忆摘要
        
        Args:
            user_id: 用户ID
            topic: 可选的主题过滤
            limit: 包含在摘要中的记忆数量
            
        Returns:
            包含摘要的字典
        """
        # 获取所有记忆
        all_memories = self.index.get_all_memories(user_id)
        
        if not all_memories:
            return {
                "topic": topic,
                "summary": "No memories found for this user.",
                "memory_count": 0,
                "success": False
            }
        
        # 构建摘要提示
        if topic:
            query = f"Summarize memories related to {topic}"
        else:
            query = "Summarize all my memories"
        
        mem_texts = [mem.get('text', '') for mem in all_memories[:limit]]
        memories_str = "\n".join([f"- {text}" for text in mem_texts])
        
        prompt = f"""Please provide a summary of the following memories.

{memories_str}

Summary:"""

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are a helpful memory summarizer."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                "topic": topic,
                "summary": response,
                "memory_count": len(all_memories),
                "memories_used": len(mem_texts),
                "success": True
            }
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {
                "topic": topic,
                "summary": "Failed to generate summary.",
                "memory_count": len(all_memories),
                "success": False
            }
    
    def get_statistics(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        获取用户记忆统计信息

        Args:
            user_id: 用户ID

        Returns:
            统计信息字典
        """
        # 使用简化的查询方式，避免一次性加载所有数据
        with self.index.db._lock:
            cur = self.index.db.connection.execute(
                """
                SELECT COUNT(DISTINCT memory_id)
                FROM history
                WHERE is_deleted = 0
                """
            )
            row = cur.fetchone()
            total_memories = row[0] if row else 0

            cur = self.index.db.connection.execute(
                """
                SELECT COUNT(*)
                FROM history
                WHERE created_at >= datetime('now', '-7 days')
                """
            )
            row = cur.fetchone()
            recent_activities = row[0] if row else 0

        stats = {
            "user_id": user_id,
            "total_memories": total_memories,
            "recent_activities": recent_activities,
            "success": True
        }

        # 可以添加更多统计信息
        # 例如：按类型、时间等分组统计

        return stats
