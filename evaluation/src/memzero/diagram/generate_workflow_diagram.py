#!/usr/bin/env python3
"""
MemoryADD 工作流程图生成器

这个脚本用于生成 MemoryADD (add.py) 的各种可视化图表，包括：
- 工作流程图
- 序列图
- 架构图
- Mermaid 格式图表
"""

import graphviz
from pathlib import Path


def generate_workflow_diagram():
    """生成 MemoryADD 的主工作流程图"""
    dot = graphviz.Digraph('MemoryADD_Workflow', 
                          format='png',
                          engine='dot',
                          graph_attr={'rankdir': 'TB', 'nodesep': '0.8', 'ranksep': '0.8'})
    
    # 设置全局属性 - 使用支持中文的字体
    dot.attr('node', shape='box', fontname='Noto Sans CJK SC', fontsize='10')
    dot.attr('edge', fontname='Noto Sans CJK SC', fontsize='9')
    
    # 定义节点组
    with dot.subgraph(name='cluster_init') as c:
        c.attr(label='初始化阶段', style='rounded,filled', color='lightblue', fontname='Noto Sans CJK SC')
        c.node('Start', '', shape='ellipse', style='filled', fillcolor='lightgreen')
        c.node('LoadEnv', '加载环境变量')
        c.node('InitClient', '初始化 MemoryClient\n(MEM0_API_KEY, model)')
        c.node('LoadData', '加载 JSON 数据集')
    
    with dot.subgraph(name='cluster_data') as c:
        c.attr(label='数据处理', style='rounded,filled', color='lightyellow', fontname='Noto Sans CJK SC')
        c.node('ConvLoop', '', shape='diamond', style='filled,dotted', fillcolor='white')
        c.node('GetConv', '获取当前对话\n(conversation, speakers)')
        c.node('MsgLoop', '', shape='diamond', style='filled,dotted', fillcolor='white')
        c.node('ExtractMsg', '提取消息对\n(role, content)')
    
    with dot.subgraph(name='cluster_parallel') as c:
        c.attr(label='并行处理', style='rounded,filled', color='lightcoral', fontname='Noto Sans CJK SC')
        c.node('CreateThread', '创建处理线程\n(threading.Thread)')
        c.node('ParallelExec', '线程池并行执行\n(concurrency > 1)')
        c.node('ProcessFunc', 'process_conversation()')
    
    with dot.subgraph(name='cluster_memory') as c:
        c.attr(label='记忆存储', style='rounded,filled', color='lavender', fontname='Noto Sans CJK SC')
        c.node('ParseBatch', '批量解析消息\n(batch_size=1)')
        c.node('CustomInst', '自定义指令\n(custom_instructions)')
        c.node('MemAdd', 'memory.add()\nAPI 调用')
        c.node('CheckResult', '检查结果\n(成功?)')
    
    with dot.subgraph(name='cluster_retry') as c:
        c.attr(label='错误处理', style='rounded,filled', color='lightpink', fontname='Noto Sans CJK SC')
        c.node('SaveResult', '保存结果到文件')
        c.node('Retry', '重试机制\n(自动重试)')
        c.node('Error', '错误处理\n(记录并跳过)')
    
    with dot.subgraph(name='cluster_end') as c:
        c.attr(label='结束', style='rounded,filled', color='lightgreen', fontname='Noto Sans CJK SC')
        c.node('End', '', shape='ellipse', style='filled', fillcolor='lightgreen')
    
    # 定义边
    dot.edge('Start', 'LoadEnv')
    dot.edge('LoadEnv', 'InitClient')
    dot.edge('InitClient', 'LoadData')
    dot.edge('LoadData', 'ConvLoop', label='开始遍历')
    
    dot.edge('ConvLoop', 'GetConv', label='下一个对话')
    dot.edge('GetConv', 'MsgLoop', label='开始处理')
    
    dot.edge('MsgLoop', 'ExtractMsg', label='提取消息')
    dot.edge('ExtractMsg', 'CreateThread', label='准备线程')
    dot.edge('CreateThread', 'ParallelExec')
    dot.edge('ParallelExec', 'ProcessFunc')
    dot.edge('ProcessFunc', 'ParseBatch')
    
    dot.edge('ParseBatch', 'CustomInst')
    dot.edge('CustomInst', 'MemAdd')
    dot.edge('MemAdd', 'CheckResult')
    
    dot.edge('CheckResult', 'SaveResult', label='成功')
    dot.edge('SaveResult', 'End', label='完成')
    
    dot.edge('CheckResult', 'Retry', label='失败')
    dot.edge('Retry', 'MemAdd', label='重试')
    dot.edge('Retry', 'Error', label='超过重试次数')
    dot.edge('Error', 'End')
    
    dot.edge('MsgLoop', 'End', label='无更多消息')
    dot.edge('ConvLoop', 'End', label='无更多对话')
    
    # 保存
    output_path = Path(__file__).parent / 'workflow_diagram'
    dot.render(output_path, format='png', cleanup=True)
    print(f"✓ 工作流程图已生成: {output_path}.png")


def generate_sequence_diagram():
    """生成 MemoryADD 的序列图"""
    dot = graphviz.Digraph('MemoryADD_Sequence',
                          format='png',
                          engine='dot')
    
    dot.attr('node', shape='record', fontname='Noto Sans CJK SC', fontsize='10')
    dot.attr('edge', fontname='Noto Sans CJK SC', fontsize='9')
    dot.attr(rankdir='LR')
    
    # 定义参与者
    dot.node('User', '用户')
    dot.node('Script', 'MemoryADD\n脚本')
    dot.node('MemClient', 'MemoryClient')
    dot.node('Mem0API', 'Mem0 API\n服务')
    dot.node('File', '文件系统')
    
    # 定义交互
    dot.edge('User', 'Script', '1. 运行脚本')
    dot.edge('Script', 'File', '2. 读取 JSON 文件')
    dot.edge('File', 'Script', '3. 返回数据')
    dot.edge('Script', 'MemClient', '4. 创建客户端')
    dot.edge('Script', 'Script', '5. 创建线程池')
    dot.edge('Script', 'Script', '6. 遍历对话')
    dot.edge('Script', 'MemClient', '7. 调用 add()\n(role, content)')
    dot.edge('MemClient', 'Mem0API', '8. HTTP POST 请求')
    dot.edge('Mem0API', 'MemClient', '9. 返回结果')
    dot.edge('MemClient', 'Script', '10. 结果处理')
    dot.edge('Script', 'File', '11. 保存到文件')
    dot.edge('Script', 'User', '12. 完成')
    
    output_path = Path(__file__).parent / 'sequence_diagram'
    dot.render(output_path, format='png', cleanup=True)
    print(f"✓ 序列图已生成: {output_path}.png")


def generate_architecture_diagram():
    """生成 MemoryADD 的架构图"""
    dot = graphviz.Digraph('MemoryADD_Architecture',
                          format='png',
                          engine='dot')
    # 设置全局属性 - 使用支持中文的字体
    dot.attr('node', fontname='Noto Sans CJK SC')
    dot.attr('edge', fontname='Noto Sans CJK SC')
    dot.attr(rankdir='TB', nodesep='0.6')
    

    # 输入层
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入层', style='filled', color='lightblue', fontname='Noto Sans CJK SC')
        c.node('JSONData', 'JSON 数据集')
    
    # 处理层
    with dot.subgraph(name='cluster_process') as c:
        c.attr(label='处理层', style='filled', color='lightyellow', fontname='Noto Sans CJK SC')
        c.node('Parser', '数据解析器')
        c.node('Batch', '批处理器')
        c.node('ThreadPool', '线程池')
    
    # 服务层
    with dot.subgraph(name='cluster_service') as c:
        c.attr(label='服务层', style='filled', color='lightgreen', fontname='Noto Sans CJK SC')
        c.node('MemClient', 'MemoryClient\n(mem0 库)')
        c.node('Prompter', '指令生成器')
    
    # API 层
    with dot.subgraph(name='cluster_api') as c:
        c.attr(label='API 层', style='filled', color='lavender', fontname='Noto Sans CJK SC')
        c.node('MemAPI', 'Mem0 API\n(ENDPOINT)')
    
    # 输出层
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层', style='filled', color='lightcoral', fontname='Noto Sans CJK SC')
        c.node('Results', '结果文件\n(JSON)')
        c.node('Logs', '日志输出')
    
    # 配置
    with dot.subgraph(name='cluster_config') as c:
        c.attr(label='配置', style='filled', color='lightgrey', fontname='Noto Sans CJK SC')
        c.node('Config', '环境变量\n(API_KEY, model)')
        c.node('Settings', '参数\n(batch_size, custom_instructions)')
    
    # 定义边
    dot.edge('JSONData', 'Parser')
    dot.edge('Config', 'MemClient')
    dot.edge('Settings', 'Batch')
    dot.edge('Parser', 'ThreadPool')
    dot.edge('ThreadPool', 'Batch')
    dot.edge('Batch', 'Prompter')
    dot.edge('Prompter', 'MemClient')
    dot.edge('MemClient', 'MemAPI')
    dot.edge('MemAPI', 'MemClient', label='返回')
    dot.edge('MemClient', 'Results')
    dot.edge('Batch', 'Logs')
    
    output_path = Path(__file__).parent / 'architecture_diagram'
    dot.render(output_path, format='png', cleanup=True)
    print(f"✓ 架构图已生成: {output_path}.png")


def generate_mermaid_diagram():
    """生成 Mermaid 格式的流程图"""
    mermaid_code = '''# MemoryADD Mermaid 流程图

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
'''
    
    output_path = Path(__file__).parent / 'mermaid_diagram.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print(f"✓ Mermaid 流程图已生成: {output_path}")


def main():
    """生成所有图表"""
    print("=" * 60)
    print("MemoryADD 工作流程图生成器")
    print("=" * 60)
    
    try:
        generate_workflow_diagram()
        generate_sequence_diagram()
        generate_architecture_diagram()
        generate_mermaid_diagram()
        
        print("\n" + "=" * 60)
        print("✓ 所有图表生成完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - workflow_diagram.png       (主工作流程图)")
        print("  - sequence_diagram.png       (序列图)")
        print("  - architecture_diagram.png   (架构图)")
        print("  - mermaid_diagram.md         (Mermaid 代码)")
        print("\n提示: 访问 https://mermaid.live/ 在线渲染 Mermaid 图表")
        
    except Exception as e:
        print(f"✗ 生成图表时出错: {e}")
        print("请确保已安装 graphviz:")
        print("  pip install graphviz")
        print("  # 同时安装系统级 graphviz:")
        print("  # Ubuntu/Debian: sudo apt-get install graphviz")
        print("  # macOS: brew install graphviz")


if __name__ == '__main__':
    main()


