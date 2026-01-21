# Ollama JSON 解析错误 - 问题分析与解决方案

## 问题描述

```
ollama._types.ResponseError: error parsing tool call: raw='{"action":"update","content":"User had a scary road trip accident but son is okay...{"filter":"memories","limit":5,"offset":0,"query":"running"}', err=invalid character 'f' after object key:value pair (status code: -1)
```

### 根本原因

这是一个 JSON 格式错误，由以下几个因素造成：

1. **工具响应格式混乱**：LLM（gpt-oss:20b）生成的工具调用包含了多个 JSON 对象，但没有正确的分隔符
   - 第一个对象：`{"action":"update","content":"..."}`
   - 第二个对象：`{"filter":"memories","limit":5,...}` 
   - 这两个对象被连接在一起，形成了非法的 JSON

2. **上下文溢出**：Ollama 模型生成工具调用时可能因为以下原因生成了不规范的输出：
   - 系统提示词过长
   - 检索到的记忆过多
   - 工具定义过于复杂
   - 模型的 token 限制被触发

3. **ReAct Agent 的工具绑定问题**：LangChain 的 `create_react_agent` 在与 Ollama 交互时，对工具响应的验证不够严格

## 提供的解决方案

### 方案 1：改进原始代码（lgm.py）

已在 `lgm.py` 中实现：

1. **增强系统提示**：改进提示词，明确告诉模型返回有效的 JSON
   ```python
   "IMPORTANT TOOL CALL GUIDELINES:\n- Always return valid JSON for tool parameters\n"
   ```

2. **重试机制**：添加自动重试，捕获 JSON 解析错误
   ```python
   max_retries = 2
   for attempt in range(max_retries):
       try:
           result = self.agent.invoke(...)
       except Exception as e:
           if "error parsing tool call" in error_msg:
               # 重试
   ```

3. **优雅降级**：当工具调用失败时，返回空结果而不是崩溃
   ```python
   return "", t2 - t1  # 空结果而非异常
   ```

4. **错误处理**：在循环中捕获每个操作的错误，继续处理其他项
   ```python
   try:
       agent1.add_memory(message, config)
   except Exception as e:
       logger.error(f"Error: {e}")
       continue  # 继续处理下一个
   ```

### 方案 2：简化实现（lgm_alternative.py）

创建了 `lgm_alternative.py`，完全避免了 ReAct agent 和工具调用：

**关键改进**：

1. **不使用 ReAct agent**：避免了 LangChain 的工具绑定问题
2. **简单的内存存储**：使用列表而非分布式存储
3. **基于字符串的搜索**：使用简单的关键字匹配而非向量搜索
4. **直接 LLM 调用**：跳过工具调用，直接使用 LLM 进行答案生成

**内存管理流程**：
```python
agent1 = SimpleLangMem()
agent1.add_memory(message)        # 简单存储
response, time = agent1.search_memory(query)  # 基于关键字搜索
```

## 使用建议

### 如果要继续使用原始方案 (lgm.py)

1. 运行现有代码，它现在有重试机制和错误处理
2. 查看日志输出，看是否还有 JSON 解析错误
3. 如果仍然有错误，考虑以下优化：
   - 减少检索的记忆数量：`limit=1` 已设置
   - 缩短系统提示词：已优化
   - 增加重试次数：更改 `max_retries = 3`

### 推荐使用方案 (lgm_alternative.py)

这是一个更加健壮的实现，适合在以下场景使用：

1. **需要稳定性**：简化实现意味着更少的依赖和更少的失败点
2. **调试困难**：不需要理解 LangChain 和 Ollama 的复杂交互
3. **快速原型**：快速验证核心逻辑而无需复杂的工具调用机制

运行方式：
```bash
python lgm_alternative.py
```

## 配置建议

### 对于 Ollama 模型调优

在 `lgm.py` 或 `lgm_alternative.py` 中：

```python
model_langmem = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    # 可选：添加以下参数
    top_p=0.95,
    top_k=40,
)
```

### Token 预算优化

当前设置：
- 系统开销预留：30% (约 9800 tokens)
- 用户输入预留：70% * 70% = 49% (约 16000 tokens)
- 内存部分：约 800 字符 (约 200 tokens)

如果仍然遇到问题，可以进一步减少：

```python
max_total_chars = 400  # 从 800 降低到 400
limit = 1  # 从 1 降低（已经很低）
```

## 日志诊断

查看生成的日志文件：

```bash
tail -f logs/lgm.log          # 原始方案
tail -f logs/lgm_alternative.log  # 替代方案
```

关键日志消息：
- `Input truncated...` - 表示 token 被截断
- `Attempt X failed...` - 表示重试
- `error parsing tool call` - 表示 JSON 解析错误

## 性能对比

| 方面 | lgm.py (ReAct) | lgm_alternative.py (简化) |
|------|----------------|---------------------------|
| 复杂度 | 高 | 低 |
| 内存检索质量 | 使用向量语义搜索 | 基于关键字 |
| 错误处理 | 有重试机制 | 简单的异常捕获 |
| 稳定性 | 中等（需要重试） | 高（直接调用） |
| 速度 | 较慢 | 快速 |
| 依赖 | LangChain, LangGraph | 仅 LangChain_Ollama |

## 总结

原始的 `lgm.py` 已增强了错误处理和重试机制，应该能够处理大多数 JSON 解析错误。

如果问题持续存在，`lgm_alternative.py` 提供了一个更简单、更稳定的替代方案，牺牲了一些高级功能（向量搜索、分布式存储）来换取稳定性。
