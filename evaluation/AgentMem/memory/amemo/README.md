# AMemo 模块化记忆系统

AMemo 是一个模块化的智能记忆系统，采用**双重编码**和**双通道混合检索**架构。

## 系统架构

系统分为四个核心模块：

```
amemo/
├── __init__.py          # 模块初始化
├── index.py             # 索引模块 - AMemoryIndex
├── add.py               # 添加模块 - MemoryAdder
├── search.py            # 检索模块 - MemorySearcher
├── response.py          # 响应模块 - MemoryResponder
├── memory.py            # 主模块 - AMemo (整合所有子模块)
├── prompts.py           # 提示词模板
├── test_amemo.py        # 测试脚本
└── README.md            # 本文档
```

## 四大核心模块

### 1. Index (索引模块) - `index.py`

**功能：** 内存创建、向量存储、索引管理和历史记录

**核心类：** `AMemoryIndex`

**主要方法：**
- `create_memory()` - 创建并索引单个内存
- `get_memory()` - 获取特定内存
- `get_all_memories()` - 获取所有内存
- `update_memory()` - 更新内存内容
- `delete_memory()` - 删除单个内存
- `delete_all_memories()` - 批量删除

**创新点：**
- 向量化存储文本内容
- 自动生成唯一ID和时间戳
- 维护操作历史记录

### 2. Add (添加模块) - `add.py`

**功能：** 解析消息、创建内存、符号提取和图存储

**核心类：** `MemoryAdder`

**主要方法：**
- `add_messages()` - 添加消息列表到记忆系统
- `add()` - 添加单个文本（兼容方法）
- `_extract_and_store_symbols()` - 提取符号信息并存入图数据库

**创新点：**
- 双重编码：文本向量 + 符号提取
- 使用 LLM 提取实体、关系和时间上下文
- 将符号信息存储到图数据库

**提取的信息：**
- Entities: 主要实体（人物、项目、地点、日期）
- Core Relationship: 核心关系或动作
- Time Context: 时间上下文

### 3. Search (检索模块) - `search.py`

**功能：** 双通道混合检索：向量检索 + 符号过滤/重排序

**核心类：** `MemorySearcher`

**主要方法：**
- `search()` - 执行双通道混合检索
- `_vector_search()` - 向量检索（语义通道）
- `_re_rank_memories()` - LLM 重排序（符号通道）
- `_process_search_result()` - 处理搜索结果

**创新点：**
- 语义通道：基于向量相似度的初步检索
- 符号通道：基于实体、关系和时间的逻辑推理
- LLM 重排序：应用逻辑和符号约束过滤结果
- 自动回退机制：当重排序失败时返回向量检索结果

### 4. Response (响应模块) - `response.py`

**功能：** 基于检索结果生成上下文感知的响应

**核心类：** `MemoryResponder`

**主要方法：**
- `generate_response()` - 生成基于记忆的响应
- `generate_summary()` - 生成记忆摘要
- `get_statistics()` - 获取记忆统计信息
- `_build_context()` - 构建上下文字符串
- `_generate_response_with_context()` - 基于上下文生成响应

**创新点：**
- 上下文感知答案生成
- 自动引用检索到的记忆
- 支持话题过滤的摘要生成
- 提供详细的统计信息

## 使用方式

### 基础使用

```python
from AgentMem.memory.amemo import AMemo

# 初始化
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
    }
}

memory = AMemo.from_config(config)
```

### 1. 添加记忆 (Add)

```python
messages = [
    {"role": "user", "content": "The meeting is scheduled for next Monday"},
    {"role": "assistant", "content": "I'll block out time for that."},
]

memory_id = memory.add(
    messages=messages,
    user_id="user_123",
    metadata={"source": "conversation"}
)
```

### 2. 检索记忆 (Search)

```python
# 使用 Search 模块
results = memory.search(
    user_id="user_123",
    query="When is the meeting?",
    limit=5
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print(f"Reasoning: {result['rank_reasoning']}")
```

### 3. 生成响应 (Response)

```python
# 使用 Response 模块
response = memory.ask(
    user_id="user_123",
    query="When is the meeting?",
    include_context=True
)

print(f"Answer: {response['response']}")
print(f"Context: {response['context']}")
```

### 4. 管理记忆 (Index)

```python
# 获取所有记忆
memories = memory.get_all(user_id="user_123")

# 获取历史
history = memory.history(user_id="user_123", limit=10)

# 更新记忆
memory.update(user_id="user_123", memory_id="mem_001", new_text="...")

# 删除记忆
memory.delete(user_id="user_123", memory_id="mem_001")
```

## 高级功能

### 生成摘要

```python
summary = memory.summarize(
    user_id="user_123",
    topic="meetings",
    limit=10
)

print(summary['summary'])
```

### 获取统计

```python
stats = memory.get_stats(user_id="user_123")
print(f"Total memories: {stats['total_memories']}")
```

### 直接访问子模块

```python
# 获取索引模块
index = memory.get_index()
mem_id = index.create_memory("text", None, {"user": "user_123"})

# 获取添加模块
adder = memory.get_adder()
adder.add_messages(messages, "user_123")

# 获取检索模块
searcher = memory.get_searcher()
results = searcher.search("user_123", "query", limit=5)

# 获取响应模块
responder = memory.get_responder()
response = responder.generate_response("user_123", "question")
```

## 运行测试

```bash
cd /root/nfs/hmj/proj/mem0/evaluation/AgentMem/memory/amemo
python test_amemo.py
```

## 系统特点

### 创新点总结

1. **双重编码 (Dual Encoding)**
   - 文本向量化存储
   - 符号化信息提取（实体、关系、时间）

2. **双通道混合检索 (Dual-Channel Retrieval)**
   - 语义通道：向量相似度检索
   - 符号通道：逻辑推理和重排序

3. **上下文感知响应 (Context-Aware Response)**
   - 基于检索记忆生成答案
   - 自动引用相关记忆
   - 摘要和统计功能

### 模块化优势

- **职责分离**：每个模块负责单一功能
- **易于维护**：修改一个模块不影响其他模块
- **可扩展性**：可以轻松替换或增强单个模块
- **可测试性**：每个模块可独立测试

## 配置选项

### LLM 配置

```python
"llm": {
    "provider": "vllm",  # 或 "openai", "anthropic"
    "config": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0,
        "max_tokens": 2000,
    }
}
```

### 嵌入模型配置

```python
"embedder": {
    "provider": "huggingface",
    "config": {
        "model": "all-MiniLM-L6-v2"
    }
}
```

### 向量存储配置

```python
"vector_store": {
    "provider": "qdrant",  # 或 "faiss", "chromadb"
    "config": {
        "collection_name": "memories",
        "host": "localhost",
        "port": 6333
    }
}
```

### 图存储配置（可选）

```python
"graph_store": {
    "provider": "neo4j",  # 或 "memgraph", "neptune"
    "config": {
        "url": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
}
```

## 向后兼容

系统保留了 `Memory` 类名作为 `AMemo` 的别名，确保与旧代码的兼容性：

```python
from AgentMem.memory.amemo import Memory  # 旧代码仍然有效

memory = Memory.from_config(config)
```

## 许可证

参见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！
