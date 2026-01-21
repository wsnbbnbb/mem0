# MemoryADD - 记忆添加工具

## 概述

`MemoryADD` 是一个用于将对话数据批量添加到 Mem0 记忆系统的工具。它支持从 JSON 文件加载数据，以批处理和并行的方式将对话内容转换为结构化记忆并存储到 Mem0 平台。

## 功能特性

- **批量处理**: 支持批量处理对话消息，提高效率
- **并行执行**: 使用多线程和多进程池实现并行数据处理
- **双视角记忆**: 从对话中分别提取两个说话者的记忆
- **图记忆支持**: 可选启用图记忆功能 (`enable_graph`)
- **自定义记忆指令**: 支持自定义记忆生成的格式和内容要求
- **错误重试**: 内置重试机制，增强系统健壮性
- **进度跟踪**: 使用 tqdm 显示处理进度

## 依赖环境

### Python 包依赖

```bash
python-dotenv
tqdm
mem0
```

### 环境变量

需要在 `.env` 文件中配置以下环境变量：

```env
MEM0_API_KEY=your_api_key_here
MEM0_ORGANIZATION_ID=your_organization_id_here
MEM0_PROJECT_ID=your_project_id_here
```

## 数据格式

### 输入 JSON 格式

输入数据应为 JSON 数组，每项包含一个对话记录：

```json
[
  {
    "conversation": {
      "speaker_a": "SpeakerA_Name",
      "speaker_b": "SpeakerB_Name",
      "chat_1": [
        {
          "speaker": "SpeakerA_Name",
          "text": "消息内容"
        },
        {
          "speaker": "SpeakerB_Name",
          "text": "回复内容"
        }
      ],
      "chat_1_date_time": "2025-01-15T10:30:00",
      "chat_2": [...],
      "chat_2_date_time": "2025-01-16T14:00:00"
    }
  }
]
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `conversation` | Object | 对话数据对象 |
| `speaker_a` | String | 说话者 A 的名称 |
| `speaker_b` | String | 说话者 B 的名称 |
| `chat_N` | Array | 第 N 轮对话的消息数组 |
| `chat_N_date_time` | String | 第 N 轮对话的时间戳 |

## 使用方法

### 基本用法

```python
from memzero.add import MemoryADD

# 初始化 MemoryADD 对象
memory_adder = MemoryADD(
    data_path="path/to/your/data.json",
    batch_size=2,
    is_graph=False
)

# 处理所有对话
memory_adder.process_all_conversations(max_workers=10)
```

### 类初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_path` | str | None | JSON 数据文件路径 |
| `batch_size` | int | 2 | 每批处理的消息数量 |
| `is_graph` | bool | False | 是否启用图记忆功能 |

### 主要方法

#### `load_data()`

加载数据文件内容。

```python
data = memory_adder.load_data()
```

#### `add_memory(user_id, message, metadata, retries=3)`

向 Mem0 系统添加单条记忆。

**参数：**
- `user_id` (str): 用户唯一标识
- `message` (str/list): 要添加的消息内容
- `metadata` (dict): 元数据，如时间戳
- `retries` (int): 重试次数，默认为 3

#### `add_memories_for_speaker(speaker, messages, timestamp, desc)`

为特定说话者批量添加记忆。

**参数：**
- `speaker` (str): 说话者的用户 ID
- `messages` (list): 消息列表
- `timestamp` (str): 时间戳
- `desc` (str): 进度条描述

#### `process_conversation(item, idx)`

处理单个对话记录。

**参数：**
- `item` (dict): 包含对话数据的字典
- `idx` (int): 对话的索引

**处理流程：**
1. 为两个说话者创建用户 ID（格式：`{speaker}_{idx}`）
2. 删除两个用户的旧记忆
3. 遍历对话中的每个聊天记录
4. 为每个说话者构建消息列表
5. 使用多线程并行添加记忆

#### `process_all_conversations(max_workers=10)`

处理 JSON 文件中的所有对话记录。

**参数：**
- `max_workers` (int): 线程池最大工作线程数，默认为 10

## 自定义记忆指令

系统使用自定义指令来生成高质量的记忆。默认指令包含以下要点：

1. **自包含的记忆**: 每条记忆包含完整上下文（姓名、详细信息、情感状态等）
2. **个人叙事**: 专注于身份、家庭计划、创意爱好、心理健康等
3. **具体细节**: 包含时间框架、具体活动名称、情感背景
4. **仅提取用户消息**: 不包含助手回复
5. **段落格式**: 使用清晰的叙事结构

可以通过修改 `custom_instructions` 变量来自定义记忆生成规则。

## 工作流程

```
1. 初始化 MemoryADD
   ↓
2. 加载 JSON 数据
   ↓
3. 创建线程池 (max_workers)
   ↓
4. 遍历每个对话记录
   ↓
5. 处理单个对话:
   - 删除旧记忆
   - 解析对话消息
   - 构建两个说话者的消息视图
   - 双线程并行添加记忆
   ↓
6. 所有对话处理完成
```

## 注意事项

1. **API 限流**: 并发过高可能导致 API 限流，建议根据实际情况调整 `max_workers`
2. **数据量**: 大数据集处理可能需要较长时间，建议分批处理
3. **错误处理**: 内置重试机制最多重试 3 次，失败会抛出异常
4. **用户 ID**: 每个对话的说话者都会获得唯一的用户 ID（格式：`{speaker}_{idx}`）
5. **内存消耗**: 批处理大小 (`batch_size`) 会影响内存使用，建议根据数据规模调整

## 示例

### 完整示例

```python
import json
from memzero.add import MemoryADD

# 示例数据
sample_data = [
    {
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "chat_1": [
                {"speaker": "Alice", "text": "你好，最近怎么样？"},
                {"speaker": "Bob", "text": "我很好，正在准备考试"}
            ],
            "chat_1_date_time": "2025-01-15T10:30:00"
        }
    }
]

# 保存为 JSON 文件
with open("sample_data.json", "w") as f:
    json.dump(sample_data, f, indent=2)

# 初始化并处理
memory_adder = MemoryADD(
    data_path="sample_data.json",
    batch_size=2,
    is_graph=True
)

memory_adder.process_all_conversations(max_workers=5)
```

## 故障排除

### 问题：API 调用失败

**解决方案**: 
- 检查 `.env` 文件中的 API 密钥配置
- 确认网络连接正常
- 检查 API 配额是否用尽

### 问题：处理速度慢

**解决方案**:
- 增加 `max_workers` 值（注意不要超过 API 限流阈值）
- 调整 `batch_size` 找到最佳平衡点
- 使用较小的数据集进行测试

## 相关文件

- `search.py`: 记忆搜索工具
- `add.py`: 记忆添加工具（当前文件）

## 许可证

请参考项目根目录的 LICENSE 文件。

## 联系方式

如有问题或建议，请通过项目 Issues 反馈。
