# MemoryADD Mermaid 流程图

```mermaid
flowchart TB
    Start([开始]) --> LoadEnv[加载环境变量]
    LoadEnv --> InitClient[初始化 MemoryClient]
    InitClient --> LoadData[加载 JSON 数据]
    
    LoadData --> ConvLoop{遍历对话}
    ConvLoop --> NextConv[获取对话和说话者]
    NextConv --> MsgLoop{遍历消息}
    
    MsgLoop --> ExtractMsg[提取消息对]
    ExtractMsg --> CreateThread[创建处理线程]
    CreateThread --> Parallel{并发处理?}
    
    Parallel -->|是| ThreadPool[线程池执行]
    Parallel -->|否| Single[单线程处理]
    
    ThreadPool --> ProcessConv[process_conversation]
    Single --> ProcessConv
    
    ProcessConv --> ParseBatch[批量解析消息]
    ParseBatch --> CustomInst[应用自定义指令]
    CustomInst --> AddMemory[调用 memory.add]
    
    AddMemory --> APICall[API 请求]
    APICall --> CheckSuccess{成功?}
    
    CheckSuccess -->|是| SaveResult[保存结果]
    CheckSuccess -->|否| Retry{重试次数<3?}
    
    Retry -->|是| APICall
    Retry -->|否| LogError[记录错误]
    
    SaveResult --> MoreMsg{更多消息?}
    MoreMsg -->|是| ExtractMsg
    MoreMsg -->|否| MoreConv{更多对话?}
    
    LogError --> MoreConv
    
    MoreConv -->|是| NextConv
    MoreConv -->|否| End([完成])
    
    style Start fill:#90EE90
    style End fill:#90EE90
    style APICall fill:#FFB6C1
    style SaveResult fill:#98FB98
```


## 如何使用
1. 将以上代码复制到支持 Mermaid 的编辑器中（如 GitHub、Typora 等）
2. 或访问 https://mermaid.live/ 在线渲染
3. 可以导出为 PNG、SVG 等格式
