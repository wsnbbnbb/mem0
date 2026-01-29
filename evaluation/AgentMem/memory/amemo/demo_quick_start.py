#!/usr/bin/env python
"""
AMemo 快速开始示例
演示模块化系统的基本用法
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from AgentMem.memory.amemo.memory import AMemo


def quick_demo():
    """快速演示"""
    print("=" * 70)
    print("🚀 AMemo 模块化系统 - 快速开始")
    print("=" * 70)
    
    # 1. 初始化
    print("\n[1/4] 初始化系统...")
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
    }
    memory = AMemo.from_config(config)
    print("✓ 系统初始化完成")
    
    # 2. 添加记忆
    print("\n[2/4] 添加记忆...")
    messages = [
        {"role": "user", "content": "项目截止日期是下周五"},
        {"role": "user", "content": "会议安排在周一上午9点"},
    ]
    mem_id = memory.add(messages, user_id="demo_user")
    print(f"✓ 已添加 {len(messages)} 条记忆")
    
    # 3. 检索记忆
    print("\n[3/4] 检索记忆...")
    results = memory.search("demo_user", "什么时候截止?", limit=2)
    print(f"✓ 找到 {len(results)} 条相关记忆")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r.get('score', 0):.3f}] {r.get('text', '')[:50]}...")
    
    # 4. 生成响应
    print("\n[4/4] 生成响应...")
    response = memory.ask("demo_user", "什么时候截止?", include_context=False)
    print(f"✓ 系统回答: {response.get('response', '')[:100]}...")
    
    print("\n" + "=" * 70)
    print("✅ 快速演示完成！")
    print("=" * 70)


def modular_demo():
    """演示如何访问各个子模块"""
    print("\n" + "=" * 70)
    print("🔧 模块化访问演示")
    print("=" * 70)
    
    config = {
        "llm": {
            "provider": "vllm",
            "config": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "vllm_base_url": "http://localhost:8000/v1",
            }
        },
        "embedder": {
            "provider": "huggingface", 
            "config": {
                "model": "all-MiniLM-L6-v2"
            }
        },
    }
    
    memory = AMemo.from_config(config)
    user_id = "modular_user"
    
    # 直接使用子模块
    print("\n--- 模块 1: Index ---")
    index = memory.get_index()
    mem_id = index.create_memory(
        "这是一条测试记忆",
        None,
        {"user_id": user_id}
    )
    print(f"创建的记忆ID: {mem_id}")
    
    print("\n--- 模块 2: Add ---")
    adder = memory.get_adder()
    msg_id = adder.add_messages(
        [{"role": "user", "content": "测试添加功能"}],
        user_id
    )
    print(f"添加的消息ID: {msg_id}")
    
    print("\n--- 模块 3: Search ---")
    searcher = memory.get_searcher()
    results = searcher.search(user_id, "测试")
    print(f"检索结果数: {len(results)}")
    
    print("\n--- 模块 4: Response ---")
    responder = memory.get_responder()
    stats = responder.get_statistics(user_id)
    print(f"统计信息: {stats}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    quick_demo()
    modular_demo()
