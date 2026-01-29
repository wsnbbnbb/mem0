"""
AMemo 提示词定义
包含符号提取和重排序验证的提示词模板
"""

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
