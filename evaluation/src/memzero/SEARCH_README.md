# MemorySearch - 记忆搜索与问答工具

## 概述

`MemorySearch` 是一个用于从 Mem0 记忆系统中搜索记忆并基于搜索结果回答问题的工具。它支持语义搜索和图记忆搜索，能够基于两个对话参与者的记忆内容来回答相关问题。

## 功能特性

- **记忆搜索**: 从 Mem0 系统中搜索相关的记忆内容
- **双视角搜索**: 同时搜索两个对话参与者的记忆
- **图记忆支持**: 支持图记忆模式，返回实体关系网络
- **智能问答**: 使用 OpenAI API 基于搜索到的记忆生成答案
- **并行处理**: 支持并行处理多个问题，提高效率
- **实时保存**: 每处理一个问题后即时保存结果
- **性能跟踪**: 记录搜索时间和响应时间
- **错误重试**: 内置重试机制，增强系统健壮性
- **进度跟踪**: 使用 tqdm 显示处理进度

## 依赖环境

### Python 包依赖

```bash
python-dotenv
jinja2
openai
tqdm
mem0
```

### 环境变量

需要在 `.env` 文件中配置以下环境变量：

```env
MEM0_API_KEY=your_api_key_here
MEM0_ORGANIZATION_ID=your_organization_id_here
MEM0_PROJECT_ID=your_project_id_here
MODEL=gpt-4  # 或其他 OpenAI 模型
```

## 数据格式

### 输入 JSON 格式

输入数据应为 JSON 数组，每项包含对话数据和问答数据：

```json
[
  {
    "conversation": {
      "speaker_a": "SpeakerA_Name",
      "speaker_b": "SpeakerB_Name",
      "chat_1": [...],
      "chat_1_date_time": "2025-01-15T10:30:00"
    },
    "qa": [
      {
        "question": "Alice 最近的职业规划是什么？",
        "answer": "Alice 最近计划...",
        "category": 1,
        "evidence": ["chat_1", "chat_2"],
        "adversarial_answer": "Bob 的工作计划是..."
      }
    ]
  }
]
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `conversation` | Object | 是 | 对话数据对象（同 MemoryADD） |
| `qa` | Array | 是 | 问答数据数组 |
| `question` | String | 是 | 问题内容 |
| `answer` | String | 是 | 参考答案 |
| `category` | Integer | 否 | 问题类别 |
| `evidence` | Array | 否 | 证据来源（聊天记录编号） |
| `adversarial_answer` | String | 否 | 对抗性答案（用于测试） |

## 使用方法

### 基本用法

```python
from memzero.search import MemorySearch

# 初始化 MemorySearch 对象
memory_search = MemorySearch(
    output_path="results.json",
    top_k=10,
    filter_memories=False,
    is_graph=False
)

# 处理数据文件
memory_search.process_data_file("path/to/your/data.json")
```

### 类初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_path` | str | "results.json" | 结果输出文件路径 |
| `top_k` | int | 10 | 搜索返回的记忆数量 |
| `filter_memories` | bool | False | 是否过滤记忆 |
| `is_graph` | bool | False | 是否启用图记忆搜索 |

### 主要方法

#### `search_memory(user_id, query, max_retries=3, retry_delay=1)`

搜索指定用户的记忆。

**参数：**
- `user_id` (str): 用户唯一标识
- `query` (str): 搜索查询
- `max_retries` (int): 最大重试次数，默认为 3
- `retry_delay` (int): 重试延迟（秒），默认为 1

**返回：**
- `semantic_memories` (list): 语义记忆列表
- `graph_memories` (list): 图记忆列表（仅图模式）
- `query_time` (float): 查询耗时

**示例：**
```python
memories, graph_memories, query_time = memory_search.search_memory(
    user_id="Alice_0",
    query="Alice 的职业规划是什么？"
)
```

#### `answer_question(speaker_1_user_id, speaker_2_user_id, question, answer, category)`

基于两个说话者的记忆回答问题。

**参数：**
- `speaker_1_user_id` (str): 说话者 1 的用户 ID
- `speaker_2_user_id` (str): 说话者 2 的用户 ID
- `question` (str): 问题内容
- `answer` (str): 参考答案
- `category` (int): 问题类别

**返回：**
包含以下内容的元组：
- `response` (str): 生成的答案
- `speaker_1_memories` (list): 说话者 1 的记忆
- `speaker_2_memories` (list): 说话者 2 的记忆
- `speaker_1_memory_time` (float): 说话者 1 记忆搜索时间
- `speaker_2_memory_time` (float): 说话者 2 记忆搜索时间
- `speaker_1_graph_memories` (list): 说话者 1 的图记忆
- `speaker_2_graph_memories` (list): 说话者 2 的图记忆
- `response_time` (float): 答案生成时间

#### `process_question(val, speaker_a_user_id, speaker_b_user_id)`

处理单个问题。

**参数：**
- `val` (dict): 包含问题详情的字典
- `speaker_a_user_id` (str): 说话者 A 的用户 ID
- `speaker_b_user_id` (str): 说话者 B 的用户 ID

**返回：**
包含所有处理结果的字典

#### `process_data_file(file_path)`

处理包含问题和对话数据的 JSON 文件。

**参数：**
- `file_path` (str): 数据文件路径

**处理流程：**
1. 加载 JSON 数据
2. 遍历每个对话记录
3. 为每个说话者创建用户 ID
4. 处理该对话的所有问题
5. 实时保存结果

#### `process_questions_parallel(qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1)`

并行处理问题列表。

**参数：**
- `qa_list` (list): 问题列表
- `speaker_a_user_id` (str): 说话者 A 的用户 ID
- `speaker_b_user_id` (str): 说话者 B 的用户 ID
- `max_workers` (int): 并行工作线程数，默认为 1

**返回：**
结果列表

## 结果格式

### 输出 JSON 格式

```json
{
  "0": [
    {
      "question": "Alice 最近的职业规划是什么？",
      "answer": "Alice 最近计划...",
      "category": 1,
      "evidence": ["chat_1", "chat_2"],
      "response": "根据记忆，Alice 的职业规划是...",
      "adversarial_answer": "Bob 的工作计划是...",
      "speaker_1_memories": [
        {
          "memory": "记忆内容",
          "timestamp": "2025-01-15T10:30:00",
          "score": 0.95
        }
      ],
      "speaker_2_memories": [...],
      "num_speaker_1_memories": 5,
      "num_speaker_2_memories": 3,
      "speaker_1_memory_time": 0.23,
      "speaker_2_memory_time": 0.18,
      "speaker_1_graph_memories": [
        {
          "source": "Alice",
          "relationship": "works_at",
          "target": "Tech Company"
        }
      ],
      "speaker_2_graph_memories": [...],
      "response_time": 1.56
    }
  ]
}
```

### 结果字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | String | 原始问题 |
| `answer` | String | 参考答案 |
| `category` | Integer | 问题类别 |
| `evidence` | Array | 证据来源 |
| `response` | String | AI 生成的答案 |
| `adversarial_answer` | String | 对抗性答案（如有） |
| `speaker_1_memories` | Array | 说话者 1 的搜索记忆 |
| `speaker_2_memories` | Array | 说话者 2 的搜索记忆 |
| `num_speaker_1_memories` | Integer | 说话者 1 的记忆数量 |
| `num_speaker_2_memories` | Integer | 说话者 2 的记忆数量 |
| `speaker_1_memory_time` | Float | 说话者 1 搜索耗时（秒） |
| `speaker_2_memory_time` | Float | 说话者 2 搜索耗时（秒） |
| `speaker_1_graph_memories` | Array | 说话者 1 的图记忆（图模式） |
| `speaker_2_graph_memories` | Array | 说话者 2 的图记忆（图模式） |
| `response_time` | Float | 答案生成耗时（秒） |

## 工作模式

### 语义搜索模式 (`is_graph=False`)

- 仅返回基于语义相似度的记忆
- 使用标准的记忆搜索 API
- 适用于简单的文本问答场景

### 图记忆搜索模式 (`is_graph=True`)

- 返回语义记忆和实体关系网络
- 使用图记忆 API（`output_format="v1.1"`）
- 提供额外的上下文信息
- 适用于需要理解实体关系的复杂问答

## 提示词模板

系统使用 Jinja2 模板来生成问答提示词。根据 `is_graph` 参数选择不同的模板：

- `ANSWER_PROMPT`: 标准问答模板
- `ANSWER_PROMPT_GRAPH`: 图记忆问答模板（包含图记忆上下文）

## 工作流程

```
1. 初始化 MemorySearch
   ↓
2. 加载数据文件
   ↓
3. 遍历每个对话
   ↓
4. 遍历每个问题:
   - 搜索 Speaker 1 的记忆
   - 搜索 Speaker 2 的记忆
   - 构建提示词
   - 调用 OpenAI API 生成答案
   - 保存结果
   ↓
5. 所有问题处理完成
```

## 注意事项

1. **API 限制**: OpenAI API 有速率限制，建议控制 `max_workers` 避免超限
2. **存储成本**: 结果会实时保存，频繁写入可能影响性能
3. **记忆数量**: `top_k` 参数会影响搜索质量和响应时间
4. **图模式**: 图模式会返回更多数据，可能增加处理时间
5. **重试机制**: 内置重试机制最多重试 3 次，失败会抛出异常
6. **用户 ID**: 需要与 MemoryADD 中使用的用户 ID 格式一致 (`{speaker}_{idx}`)

## 示例

### 完整示例

```python
import json
from memzero.search import MemorySearch

# 示例数据
sample_data = [
    {
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "chat_1": [
                {"speaker": "Alice", "text": "我计划下个月换工作"},
                {"speaker": "Bob", "text": "你要去哪家公司？"}
            ],
            "chat_1_date_time": "2025-01-15T10:30:00"
        },
        "qa": [
            {
                "question": "Alice 的工作计划是什么？",
                "answer": "Alice 计划下个月换工作",
                "category": 1,
                "evidence": ["chat_1"]
            }
        ]
    }
]

# 保存为 JSON 文件
with open("sample_data.json", "w") as f:
    json.dump(sample_data, f, indent=2)

# 使用标准模式
memory_search = MemorySearch(
    output_path="standard_results.json",
    top_k=10,
    is_graph=False
)

memory_search.process_data_file("sample_data.json")

# 使用图模式
memory_search_graph = MemorySearch(
    output_path="graph_results.json",
    top_k=10,
    is_graph=True
)

memory_search_graph.process_data_file("sample_data.json")
```

### 单独搜索记忆

```python
from memzero.search import MemorySearch

memory_search = MemorySearch()

# 搜索特定用户的记忆
memories, graph_memories, query_time = memory_search.search_memory(
    user_id="Alice_0",
    query="职业规划"
)

print(f"找到 {len(memories)} 条记忆")
print(f"查询耗时: {query_time:.2f} 秒")
for mem in memories:
    print(f"- {mem['timestamp']}: {mem['memory']} (分数: {mem['score']})")
```

### 回答单个问题

```python
from memzero.search import MemorySearch

memory_search = MemorySearch()

response, mem1, mem2, time1, time2, graph1, graph2, resp_time = \
    memory_search.answer_question(
        speaker_1_user_id="Alice_0",
        speaker_2_user_id="Bob_0",
        question="Alice 的工作计划是什么？",
        answer="Alice 计划换工作",
        category=1
    )

print(f"答案: {response}")
print(f"记忆数量: Speaker 1 ({len(mem1)}), Speaker 2 ({len(mem2)})")
print(f"响应时间: {resp_time:.2f} 秒")
```

## 性能优化

1. **调整 top_k**: 根据问题复杂度调整返回的记忆数量
2. **并行处理**: 使用 `process_questions_parallel` 提高吞吐量
3. **缓存结果**: 对于重复查询考虑实现缓存机制
4. **批处理**: 减少频繁的磁盘 I/O 操作
5. **模型选择**: 选择合适复杂度的 OpenAI 模型

## 故障排除

### 问题：记忆搜索返回空结果

**解决方案**:
- 检查用户 ID 是否正确（格式应为 `{speaker}_{idx}`）
- 确认该用户有记忆被添加到系统
- 检查搜索查询是否过于具体或抽象

### 问题：答案质量不佳

**解决方案**:
- 增加 `top_k` 值以获取更多上下文
- 使用图模式获取实体关系
- 调整提示词模板
- 使用更强大的 OpenAI 模型

### 问题：处理速度慢

**解决方案**:
- 减少 `top_k` 值
- 使用并行处理功能
- 考虑缓存搜索结果
- 检查网络连接质量

### 问题：API 调用失败

**解决方案**:
- 检查 `.env` 文件中的 API 密钥配置
- 确认网络连接正常
- 检查 API 配额是否用尽
- 增加 `max_retries` 和 `retry_delay` 值

## 输出分析

生成的结果文件可以用于：

1. **准确性评估**: 将生成的答案与参考答案对比
2. **记忆召回分析**: 检查检索到的记忆是否相关
3. **性能分析**: 分析搜索和响应时间
4. **类别统计**: 分析不同类别问题的表现
5. **对抗性测试**: 使用 `adversarial_answer` 进行鲁棒性测试

## 相关文件

- `add.py`: 记忆添加工具（需要先运行此工具添加记忆）
- `search.py`: 记忆搜索工具（当前文件）
- `prompts.py`: 问答提示词模板

## 许可证

请参考项目根目录的 LICENSE 文件。

## 联系方式

如有问题或建议，请通过项目 Issues 反馈。
