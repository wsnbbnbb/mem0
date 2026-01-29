#!/usr/bin/env python
"""
AMemo 模块化系统测试脚本
演示如何使用拆分后的四个核心模块
"""

from AgentMem.memory.amemo import AMemo


def main():
    """主测试函数"""
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
        "embedder": {
            "provider": "huggingface", 
            "config": {
                "model": "all-MiniLM-L6-v2"
            }
        },
        # 可选：启用图存储
        # "vector_store": {
        #     "provider": "qdrant",
        #     "config": {
        #         "collection_name": "vllm_memories", 
        #         "host": "localhost", 
        #         "port": 6333
        #     },
        # },
        # "graph_store": {
        #     "provider": "neo4j",
        #     "config": {
        #         "url": "bolt://localhost:7687",
        #         "username": "neo4j",
        #         "password": "Neo4j2025",
        #         "database": "neo4j",
        #     }
        # }
    }
    
    print("=" * 80)
    print("🧠 AMemo 模块化记忆系统测试")
    print("=" * 80)
    
    # 初始化系统
    print("\n## 🔧 1. 初始化系统")
    memory = AMemo.from_config(config)
    user_id = "test_user"
    print(f"✓ 系统初始化完成")
    print(f"  用户ID: {user_id}")
    
    # 获取各个模块（可选）
    print("\n  可用模块:")
    print(f"  - 索引模块: {type(memory.get_index()).__name__}")
    print(f"  - 添加模块: {type(memory.get_adder()).__name__}")
    print(f"  - 检索模块: {type(memory.get_searcher()).__name__}")
    print(f"  - 响应模块: {type(memory.get_responder()).__name__}")
    
    # ========== 模块 1: ADD ==========
    print("\n" + "=" * 80)
    print("## 📝 模块 1: ADD - 添加记忆")
    print("=" * 80)
    
    # 示例对话文本
    dialogue_1 = {
        "role": "user",
        "content": "Melanie: Hey Caroline, since we last chatted, I've had a lot of "
                   "things happening to me. I ran a charity race for mental health last "
                   "Saturday – it was really rewarding. Really made me think about taking "
                   "care of our minds."
    }
    
    dialogue_2 = {
        "role": "user",
        "content": "Melanie: The Q4 Report review is scheduled for next Monday. "
                   "Caroline: Perfect, I'll block out time for that."
    }
    
    messages = [dialogue_1, dialogue_2]
    
    print(f"\n要添加的消息数量: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  消息 {i}: {content}")
    
    # 添加记忆
    mem_id = memory.add(
        messages=messages, 
        user_id=user_id, 
        metadata={"timestamp": "Jan 28 2026"}
    )
    
    print(f"\n✓ 记忆添加成功!")
    print(f"  记忆ID: {mem_id}")
    
    # ========== 模块 2: INDEX ==========
    print("\n" + "=" * 80)
    print("## 📚 模块 2: INDEX - 索引管理")
    print("=" * 80)
    
    # 获取所有记忆
    all_memories = memory.get_all(user_id)
    print(f"\n总记忆数: {len(all_memories)}")
    
    # 展示记忆详情
    for i, mem in enumerate(all_memories, 1):
        print(f"\n记忆 {i}:")
        print(f"  ID: {mem.get('id', 'N/A')}")
        print(f"  数据: {str(mem.get('data', '')[:80])}...")
        print(f"  创建时间: {mem.get('created_at', 'N/A')}")
    
    # 获取历史记录
    history = memory.history(user_id, limit=3)
    print(f"\n最近的历史记录: {len(history)} 条")
    
    # ========== 模块 3: SEARCH ==========
    print("\n" + "=" * 80)
    print("## 🔎 模块 3: SEARCH - 双通道混合检索")
    print("=" * 80)
    
    queries = [
        "When is the Q4 Report review scheduled?",
        "What did Melanie do last Saturday?",
        "Tell me about Caroline's plans"
    ]
    
    for query in queries:
        print(f"\n🔎 查询: '{query}'")
        
        # 使用 Search 模块
        results = memory.search(user_id, query, limit=2)
        
        if results:
            print(f"✓ 找到 {len(results)} 条相关记忆:")
            for i, result in enumerate(results, 1):
                print(f"\n  排名 {i}:")
                print(f"    分数: {result.get('score', 'N/A')}")
                print(f"    推理: {result.get('rank_reasoning', 'N/A')}")
                print(f"    内容: {result.get('text', 'N/A')[:100]}...")
        else:
            print("✗ 未找到相关记忆")
    
    # ========== 模块 4: RESPONSE ==========
    print("\n" + "=" * 80)
    print("## 💬 模块 4: RESPONSE - 上下文感知响应生成")
    print("=" * 80)
    
    question = "When is the Q4 Report review scheduled?"
    print(f"\n❓ 问题: '{question}'")
    
    # 生成响应
    response_obj = memory.ask(user_id, question, limit=2, include_context=True)
    
    print("\n" + "-" * 80)
    print("📋 检索到的上下文:")
    print("-" * 80)
    print(response_obj.get('context', 'No context available'))
    
    print("\n" + "-" * 80)
    print("💬 系统响应:")
    print("-" * 80)
    print(response_obj.get('response', 'No response available'))
    
    # ========== 摘要和统计 ==========
    print("\n" + "=" * 80)
    print("## 📊 摘要和统计")
    print("=" * 80)
    
    # 生成摘要
    summary = memory.summarize(user_id, limit=5)
    print(f"\n📝 记忆摘要:")
    print(f"  记忆总数: {summary.get('memory_count', 0)}")
    print(f"  摘要内容:\n{summary.get('summary', 'No summary available')}")
    
    # 获取统计
    stats = memory.get_stats(user_id)
    print(f"\n📈 统计信息:")
    print(f"  用户ID: {stats.get('user_id', 'N/A')}")
    print(f"  总记忆数: {stats.get('total_memories', 0)}")
    print(f"  最近活动: {stats.get('recent_activities', 0)}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成!")
    print("=" * 80)
    
    return memory


if __name__ == "__main__":
    memory_system = main()
