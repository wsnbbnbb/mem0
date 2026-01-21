#!/usr/bin/env python3
"""
MemorySearch 工作流流程图生成器

该脚本使用 graphviz 库生成 MemorySearch 的工作流程图。
安装依赖: pip install graphviz
"""

from graphviz import Digraph


def generate_workflow_diagram(output_file='search_workflow_diagram', format='png'):
    """
    生成 MemorySearch 工作流程图

    Args:
        output_file (str): 输出文件名（不带扩展名）
        format (str): 输出格式（png, svg, pdf 等）
    """
    # 创建有向图
    dot = Digraph(comment='MemorySearch 工作流程图',
                  graph_attr={'rankdir': 'TB',
                             'fontname': 'Arial',
                             'fontsize': '12',
                             'splines': 'ortho'},
                  node_attr={'fontname': 'Arial',
                            'fontsize': '10',
                            'shape': 'box',
                            'style': 'rounded',
                            'margin': '0.15,0.1'},
                  edge_attr={'fontname': 'Arial',
                            'fontsize': '9'})

    # ==================== 初始化层 ====================
    dot.node('start', shape='oval', style='filled', fillcolor='#90EE90',
             label='开始\n初始化 MemorySearch\n设置 top_k, is_graph, filter_memories')

    dot.node('load_env', shape='parallelogram', style='filled', fillcolor='#87CEEB',
             label='加载环境变量\n(MEM0_API_KEY, MODEL等)')

    dot.node('init_clients', style='filled', fillcolor='#FFD700',
             label='初始化客户端\nMem0Client\nOpenAI Client')

    dot.node('load_prompt', shape='note', style='filled', fillcolor='#98FB98',
             label='加载提示词模板\n(ANSWER_PROMPT)')

    # ==================== 数据加载层 ====================
    dot.node('load_data', shape='parallelogram', style='filled', fillcolor='#87CEEB',
             label='加载 JSON 数据文件\n(包含对话和问题)')

    # ==================== 遍历对话层 ====================
    dot.node('iterate_conv', shape='diamond', style='filled', fillcolor='#FF69B4',
             label='遍历对话\n(for each conversation)')

    dot.node('get_speaker_info', style='filled', fillcolor='#E6E6FA',
             label='获取说话者信息\nspeaker_a, speaker_b')

    dot.node('create_user_ids', style='filled', fillcolor='#E6E6FA',
             label='创建用户 ID\nspeaker_{idx}')

    # ==================== 遍历问题层 ====================
    dot.node('iterate_qa', shape='diamond', style='filled', fillcolor='#FF69B4',
             label='遍历问题\n(for each question)')

    dot.node('extract_question', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
             label='提取问题信息\nquestion, answer\ncategory, evidence')

    # ==================== 搜索记忆层 ====================
    dot.node('search_mem_parallel', shape='diamond', style='filled', fillcolor='#FFD700',
             label='并行搜索两个用户的记忆')

    dot.node('search_speaker1', shape='parallelogram', style='filled', fillcolor='#98FB98',
             label='搜索 Speaker 1 记忆\n(search_memory)')

    dot.node('search_speaker2', shape='parallelogram', style='filled', fillcolor='#98FB98',
             label='搜索 Speaker 2 记忆\n(search_memory)')

    # ==================== 搜索内部流程 ====================
    dot.node('api_call', shape='parallelogram', style='filled', fillcolor='#87CEEB',
             label='Mem0 搜索 API\n(top_k, filter_memories)')

    dot.node('check_mode', shape='diamond', style='filled', fillcolor='#FFD700',
             label='图模式?\n(is_graph)')

    dot.node('graph_search', shape='parallelogram', style='filled', fillcolor='#DDA0DD',
             label='图记忆搜索\n(enable_graph=True\noutput_format=v1.1)')

    dot.node('semantic_search', shape='parallelogram', style='filled', fillcolor='#DDA0DD',
             label='语义搜索\n(标准搜索)')

    dot.node('check_retry', shape='diamond', style='filled', fillcolor='#DC143C',
             label='成功?')

    dot.node('wait_retry', shape='parallelogram', style='filled', fillcolor='#FFA07A',
             label='等待后重试\n(time.sleep)')

    dot.node('raise_error', shape='parallelogram', style='filled', fillcolor='#8B0000',
             label='抛出异常\n(raise error)')

    # ==================== 处理搜索结果 ====================
    dot.node('extract_memories', style='filled', fillcolor='#FFD700',
             label='提取记忆数据\nmemory, timestamp, score')

    dot.node('extract_graph', style='filled', fillcolor='#E6E6FA',
             label='提取图关系\n(source, relationship, target)\n(仅图模式)')

    dot.node('format_results', style='filled', fillcolor='#E6E6FA',
             label='格式化结果\nJSON 格式')

    # ==================== 构建提示词 ====================
    dot.node('build_prompt', style='filled', fillcolor='#FFD700',
             label='构建提示词\n使用 Jinja2 模板\n注入记忆和问题')

    dot.node('render_template', shape='parallelogram', style='filled', fillcolor='#87CEEB',
             label='渲染模板\n(Template.render)\n插入记忆数据')

    # ==================== 生成答案 ====================
    dot.node('openai_api', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
             label='调用 OpenAI API\n(chat.completions.create)')

    dot.node('set_temperature', style='filled', fillcolor='#E6E6FA',
             label='设置参数\nmodel, temperature=0.0')

    # ==================== 保存结果 ====================
    dot.node('package_result', style='filled', fillcolor='#FFD700',
             label='打包结果\nresponse, memories\ntimes, graph_memories')

    dot.node('save_result', shape='parallelogram', style='filled', fillcolor='#98FB98',
             label='实时保存\n(json.dump\nto output_path)')

    # ==================== 循环控制 ====================
    dot.node('more_qa', shape='diamond', style='filled', fillcolor='#FF69B4',
             label='更多问题?')

    dot.node('more_conv', shape='diamond', style='filled', fillcolor='#FF69B4',
             label='更多对话?')

    # ==================== 完成 ====================
    dot.node('final_save', shape='parallelogram', style='filled', fillcolor='#98FB98',
             label='最终保存\n所有结果')

    dot.node('end', shape='oval', style='filled', fillcolor='#90EE90',
             label='完成\n结果已保存到\noutput_path')

    # ==================== 边连接 ====================
    # 主初始化流程
    dot.edge('start', 'load_env')
    dot.edge('load_env', 'init_clients')
    dot.edge('init_clients', 'load_prompt')
    dot.edge('load_prompt', 'load_data')

    # 数据加载
    dot.edge('load_data', 'iterate_conv')

    # 遍历对话
    dot.edge('iterate_conv', 'get_speaker_info')
    dot.edge('get_speaker_info', 'create_user_ids')
    dot.edge('create_user_ids', 'iterate_qa')

    # 遍历问题
    dot.edge('iterate_qa', 'extract_question')
    dot.edge('extract_question', 'search_mem_parallel')

    # 并行搜索
    dot.edge('search_mem_parallel', 'search_speaker1', label='同时执行')
    dot.edge('search_mem_parallel', 'search_speaker2', label='同时执行')

    # 搜索详细流程
    dot.edge('search_speaker1', 'check_mode')
    dot.edge('search_speaker2', 'check_mode')

    dot.edge('check_mode', 'graph_search', label='是')
    dot.edge('check_mode', 'semantic_search', label='否')

    dot.edge('graph_search', 'api_call')
    dot.edge('semantic_search', 'api_call')
    dot.edge('api_call', 'check_retry')

    dot.edge('check_retry', 'extract_memories', label='是')
    dot.edge('check_retry', 'wait_retry', label='否')
    dot.edge('wait_retry', 'api_call', label='重试')
    dot.edge('check_retry', 'raise_error', label='失败')

    # 处理搜索结果
    dot.edge('extract_memories', 'check_mode')
    dot.edge('check_mode', 'extract_graph', label='是')
    dot.edge('check_mode', 'format_results', label='否')
    dot.edge('extract_graph', 'format_results')

    # 等待两个搜索完成
    dot.edge('format_results_from_s1', 'format_results', label='Speaker 1 完成')
    dot.edge('format_results_from_s2', 'format_results', label='Speaker 2 完成')

    # 构建提示词
    dot.edge('format_results', 'build_prompt')
    dot.edge('build_prompt', 'render_template')

    # 生成答案
    dot.edge('render_template', 'set_temperature')
    dot.edge('set_temperature', 'openai_api')

    # 保存结果
    dot.edge('openai_api', 'package_result')
    dot.edge('package_result', 'save_result')

    # 循环
    dot.edge('save_result', 'more_qa')
    dot.edge('more_qa', 'extract_question', label='是')
    dot.edge('more_qa', 'more_conv', label='否')

    dot.edge('more_conv', 'get_speaker_info', label='是')
    dot.edge('more_conv', 'final_save', label='否')

    dot.edge('final_save', 'end')

    # 保存并渲染
    try:
        output_path = dot.render(output_file, format=format, cleanup=True)
        print(f"✅ 流程图已生成: {output_path}")
        print(f"💡 提示: 请在文件管理器中查看生成的 {output_file}.{format} 文件")
    except Exception as e:
        print(f"❌ 生成流程图失败: {e}")
        print(f"💡 请先安装 graphviz:")
        print(f"   - Ubuntu/Debian: sudo apt-get install graphviz")
        print(f"   - macOS: brew install graphviz")
        print(f"   - Windows: 从 https://graphviz.org/download/ 下载安装")
        print(f"   - Python: pip install graphviz")

    return dot


def generate_detailed_flow_diagram(output_file='search_detailed_diagram', format='png'):
    """
    生成详细的 MemorySearch 流程图（包含方法调用细节）

    Args:
        output_file (str): 输出文件名
        format (str): 输出格式
    """
    dot = Digraph(comment='MemorySearch 详细流程图',
                  graph_attr={'rankdir': 'TB',
                             'fontname': 'Arial',
                             'fontsize': '11',
                             'splines': 'ortho'},
                  node_attr={'fontname': 'Arial',
                            'fontsize': '9',
                            'style': 'rounded'},
                  edge_attr={'fontname': 'Arial',
                            'fontsize': '8'})

    # 设置集群
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='主流程', style='filled', color='lightyellow')

        c.node('main_start', shape='oval', style='filled', fillcolor='#90EE90',
               label='process_data_file()')
        c.node('load_json', shape='parallelogram', style='filled', fillcolor='#87CEEB',
               label='json.load(file_path)')
        c.node('main_loop_start', shape='diamond', style='filled', fillcolor='#FF69B4',
               label='for idx, item in data')
        c.node('extract_qa', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
               label='qa = item["qa"]\nconversation = item["conversation"]')
        c.node('qa_loop_start', shape='diamond', style='filled', fillcolor='#FF69B4',
               label='for question_item in qa')
        c.node('call_process_q', shape='parallelogram', style='filled', fillcolor='#98FB98',
               label='process_question(val,\nspeaker_a_user_id,\nspeaker_b_user_id)')
        c.node('save_results', shape='parallelogram', style='filled', fillcolor='#98FB98',
               label='json.dump(results, f)')
        c.node('qa_loop_end', shape='diamond', style='filled', fillcolor='#FF69B4',
               label='继续下个问题?')
        c.node('main_loop_end', shape='diamond', style='filled', fillcolor='#FF69B4',
               label='继续下个对话?')

        c.edge('main_start', 'load_json')
        c.edge('load_json', 'main_loop_start')
        c.edge('main_loop_start', 'extract_qa')
        c.edge('extract_qa', 'qa_loop_start')
        c.edge('qa_loop_start', 'call_process_q')
        c.edge('call_process_q', 'save_results')
        c.edge('save_results', 'qa_loop_end')
        c.edge('qa_loop_end', 'qa_loop_start', label='是', xlabel='back')
        c.edge('qa_loop_end', 'main_loop_end', label='否')
        c.edge('main_loop_end', 'extract_qa', label='是', xlabel='back')
        c.edge('main_loop_end', 'main_end', label='否')

    with dot.subgraph(name='cluster_process_q') as c:
        c.attr(label='process_question() 方法', style='filled', color='lightgreen')

        c.node('pq_start', shape='oval', style='filled', fillcolor='#90EE90',
               label='process_question()')
        c.node('pq_extract', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
               label='提取问题字段\nquestion, answer\ncategory, evidence\nadversarial_answer')
        c.node('call_answer_q', shape='parallelogram', style='filled', fillcolor='#DDA0DD',
               label='answer_question()\n返回多个值')
        c.node('pq_build_result', shape='parallelogram', style='filled', fillcolor='#FFD700',
               label='构建结果字典\n包含所有响应数据')
        c.node('pq_save', shape='parallelogram', style='filled', fillcolor='#98FB98',
               label='json.dump(results, f)')
        c.node('pq_return', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
               label='return result')

        c.edge('pq_start', 'pq_extract')
        c.edge('pq_extract', 'call_answer_q')
        c.edge('call_answer_q', 'pq_build_result')
        c.edge('pq_build_result', 'pq_save')
        c.edge('pq_save', 'pq_return')

    with dot.subgraph(name='cluster_answer_q') as c:
        c.attr(label='answer_question() 方法', style='filled', color='lightblue')

        c.node('aq_start', shape='oval', style='filled', fillcolor='#90EE90',
               label='answer_question()')
        c.node('aq_search_1', shape='parallelogram', style='filled', fillcolor='#87CEEB',
               label='search_memory()\nSpeaker 1')
        c.node('aq_search_2', shape='parallelogram', style='filled', fillcolor='#87CEEB',
               label='search_memory()\nSpeaker 2')
        c.node('aq_format', style='filled', fillcolor='#E6E6FA',
               label='格式化记忆数据\n时间戳 + 内容')
        c.node('aq_template', shape='parallelogram', style='filled', fillcolor='#98FB98',
               label='Template.render()\n注入变量')
        c.node('aq_openai', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
               label='OpenAI API 调用\nchat.completions.create()')
        c.node('aq_time_calc', style='filled', fillcolor='#E6E6FA',
               label='计算响应时间')
        c.node('aq_return', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
               label='return 8 个值')

        c.edge('aq_start', 'aq_search_1', label='并行')
        c.edge('aq_start', 'aq_search_2', label='并行')
        c.edge('aq_search_1', 'aq_format')
        c.edge('aq_search_2', 'aq_format')
        c.edge('aq_format', 'aq_template')
        c.edge('aq_template', 'aq_openai')
        c.edge('aq_openai', 'aq_time_calc')
        c.edge('aq_time_calc', 'aq_return')

    with dot.subgraph(name='cluster_search_mem') as c:
        c.attr(label='search_memory() 方法', style='filled', color='lightcoral')

        c.node('sm_start', shape='oval', style='filled', fillcolor='#90EE90',
               label='search_memory()')
        c.node('sm_time_start', style='filled', fillcolor='#E6E6FA',
               label='start_time = time.time()')
        c.node('sm_retry_loop', shape='diamond', style='filled', fillcolor='#FF69B4',
               label='while retries < max_retries')
        c.node('sm_check_mode', shape='diamond', style='filled', fillcolor='#FFD700',
               label='is_graph?')
        c.node('sm_graph_call', shape='parallelogram', style='filled', fillcolor='#DDA0DD',
               label='mem0_client.search()\nenable_graph=True\noutput_format=v1.1')
        c.node('sm_semantic_call', shape='parallelogram', style='filled', fillcolor='#DDA0DD',
               label='mem0_client.search()\n标准搜索')
        c.node('sm_check_success', shape='diamond', style='filled', fillcolor='#DC143C',
               label='成功?')
        c.node('sm_increment', style='filled', fillcolor='#E6E6FA',
               label='retries += 1')
        c.node('sm_sleep', shape='parallelogram', style='filled', fillcolor='#FFA07A',
               label='time.sleep(retry_delay)')
        c.node('sm_raise', shape='parallelogram', style='filled', fillcolor='#8B0000',
               label='raise error')
        c.node('sm_time_end', style='filled', fillcolor='#E6E6FA',
               label='end_time = time.time()')
        c.node('sm_extract', style='filled', fillcolor='#E6E6FA',
               label='提取记忆数据\nmemory, timestamp, score')
        c.node('sm_extract_graph', style='filled', fillcolor='#E6E6FA',
               label='提取图关系\n(如果 is_graph)')
        c.node('sm_return', shape='parallelogram', style='filled', fillcolor='#FFB6C1',
               label='return\n(semantic_memories,\ngraph_memories,\nquery_time)')

        c.edge('sm_start', 'sm_time_start')
        c.edge('sm_time_start', 'sm_retry_loop')
        c.edge('sm_retry_loop', 'sm_check_mode')
        c.edge('sm_check_mode', 'sm_graph_call', label='是')
        c.edge('sm_check_mode', 'sm_semantic_call', label='否')
        c.edge('sm_graph_call', 'sm_check_success')
        c.edge('sm_semantic_call', 'sm_check_success')
        c.edge('sm_check_success', 'sm_time_end', label='是')
        c.edge('sm_check_success', 'sm_increment', label='否')
        c.edge('sm_increment', 'sm_sleep')
        c.edge('sm_sleep', 'sm_retry_loop', label='back')
        c.edge('sm_retry_loop', 'sm_raise', label='失败')
        c.edge('sm_time_end', 'sm_extract')
        c.edge('sm_extract', 'sm_extract_graph')
        c.edge('sm_extract_graph', 'sm_return')

    # 主流程到子流程的连接
    dot.edge('main_end', 'pq_start', style='dashed', label='调用')
    dot.edge('pq_return', 'aq_start', style='dashed', label='调用')
    dot.edge('aq_return', 'sm_start', style='dashed', label='调用')

    try:
        output_path = dot.render(output_file, format=format, cleanup=True)
        print(f"✅ 详细流程图已生成: {output_path}")
    except Exception as e:
        print(f"❌ 生成详细流程图失败: {e}")

    return dot


def generate_sequence_diagram(output_file='search_sequence_diagram', format='png'):
    """
    生成时序图（展示组件交互流程）

    Args:
        output_file (str): 输出文件名
        format (str): 输出格式
    """
    dot = Digraph(comment='MemorySearch 时序图',
                  graph_attr={'rankdir': 'LR',
                             'fontname': 'Arial'},
                  node_attr={'fontname': 'Arial',
                            'shape': 'box',
                            'style': 'rounded'},
                  edge_attr={'fontname': 'Arial',
                            'fontsize': '8',
                            'labelangle': '10'})

    # 定义参与者
    dot.node('User', shape='box', style='filled', fillcolor='#E6E6FA', label='用户')
    dot.node('MemorySearch', shape='box', style='filled', fillcolor='#87CEEB', label='MemorySearch')
    dot.node('Jinja2', shape='box', style='filled', fillcolor='#FFD700', label='Jinja2 模板')
    dot.node('Mem0API', shape='box', style='filled', fillcolor='#FFA07A', label='Mem0 API')
    dot.node('OpenAI', shape='box', style='filled', fillcolor='#FFB6C1', label='OpenAI API')
    dot.node('FileSystem', shape='cylinder', style='filled', fillcolor='#98FB98', label='文件系统')

    # 创建消息节点
    messages = [
        ('u1', 'm1', '1. 初始化和加载数据'),
        ('m1', 'fs1', '2. 读取 JSON 文件'),
        ('fs1', 'm1', '3. 返回数据'),
        ('m1', 'mem0_1', '4. 搜索 Speaker 1 记忆'),
        ('m1', 'mem0_2', '5. 搜索 Speaker 2 记忆 (并行)'),
        ('mem0_1', 'm1', '6. 返回语义记忆'),
        ('mem0_2', 'm1', '7. 返回语义记忆'),
        ('mem0_1', 'm1', '8. 返回图关系 (图模式)'),
        ('mem0_2', 'm1', '9. 返回图关系 (图模式)'),
        ('m1', 'j2', '10. 渲染提示词模板'),
        ('j2', 'm1', '11. 返回格式化提示词'),
        ('m1', 'openai1', '12. 发送请求到 OpenAI'),
        ('openai1', 'm1', '13. 返回生成的答案'),
        ('m1', 'fs2', '14. 实时保存结果'),
        ('m1', 'u1', '15. 返回处理结果'),
    ]

    # 添加节点和边
    for idx, (src, dst, label) in enumerate(messages):
        src_node = f'{src}_point' if idx > 0 else 'u1'
        dst_node = f'{dst}_{idx}'

        if idx == 0:
            dot.edge(src, dst, label=label)
        else:
            dot.edge(src_node, dst_node, label=label)

        # 更新源节点为当前目的节点
        src_node = dst_node

    try:
        output_path = dot.render(output_file, format=format, cleanup=True)
        print(f"✅ 时序图已生成: {output_path}")
    except Exception as e:
        print(f"❌ 生成时序图失败: {e}")

    return dot


def generate_architecture_diagram(output_file='search_architecture_diagram', format='png'):
    """
    生成架构图

    Args:
        output_file (str): 输出文件名
        format (str): 输出格式
    """
    dot = Digraph(comment='MemorySearch 系统架构图',
                  graph_attr={'rankdir': 'TB',
                             'fontname': 'Arial',
                             'fontsize': '12'},
                  node_attr={'fontname': 'Arial',
                            'fontsize': '10',
                            'style': 'filled'},
                  edge_attr={'fontname': 'Arial',
                            'fontsize': '9'})

    # 添加子图以组织不同的层
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入层', style='filled', color='lightyellow')
        c.node('json_file', shape='folder', fillcolor='#FFD700',
               label='JSON 数据文件\n(对话 + 问题)')
        c.node('env_file', shape='folder', fillcolor='#FFD700',
               label='.env 文件\n(API 密钥)')

    with dot.subgraph(name='cluster_processing') as c:
        c.attr(label='处理层', style='filled', color='lightgreen')
        c.node('memory_search', shape='component', fillcolor='#87CEEB',
               label='MemorySearch 类\n(主控制器)')
        c.node('data_processor', shape='box', fillcolor='#DDA0DD',
               label='数据处理器\n(process_data_file)')
        c.node('qa_processor', shape='box', fillcolor='#DDA0DD',
               label='问题处理器\n(process_question)')
        c.node('search_processor', shape='box', fillcolor='#DDA0DD',
               label='搜索处理器\n(search_memory)')

    with dot.subgraph(name='cluster_template') as c:
        c.attr(label='模板层', style='filled', color='lightblue')
        c.node('jinja2', shape='component', fillcolor='#FFD700',
               label='Jinja2 模板引擎')
        c.node('answer_prompt', shape='note', fillcolor='#98FB98',
               label='ANSWER_PROMPT\n(提示词模板)')
        c.node('graph_prompt', shape='note', fillcolor='#98FB98',
               label='ANSWER_PROMPT_GRAPH\n(图模式模板)')

    with dot.subgraph(name='cluster_memory_layer') as c:
        c.attr(label='记忆层', style='filled', color='lightcoral')
        c.node('mem0_client', shape='component', fillcolor='#FF6347',
               label='Mem0Client\n(API 客户端)')
        c.node('search_api', shape='database', fillcolor='#FFB6C1',
               label='搜索 API\n(search)')
        c.node('graph_search', shape='database', fillcolor='#FFB6C1',
               label='图搜索 API\n(enable_graph=True)')

    with dot.subgraph(name='cluster_llm_layer') as c:
        c.attr(label='LLM 层', style='filled', color='lavender')
        c.node('openai_client', shape='component', fillcolor='#FFA500',
               label='OpenAI 客户端')
        c.node('chat_api', shape='database', fillcolor='#FFB6C1',
               label='聊天完成 API\n(chat.completions)')

    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出层', style='filled', color='lightgray')
        c.node('json_output', shape='folder', fillcolor='#98FB98',
               label='results.json\n(输出结果)')

    with dot.subgraph(name='cluster_external') as c:
        c.attr(label='外部服务', style='dashed', color='gray')
        c.node('mem0_cloud', shape='cloud3', fillcolor='#D3D3D3',
               label='Mem0 云服务')
        c.node('openai_service', shape='cloud3', fillcolor='#E6E6FA',
               label='OpenAI 服务')

    # 添加关系边
    # 输入到处理
    dot.edge('json_file', 'memory_search', label='加载')
    dot.edge('env_file', 'memory_search', label='配置')
    dot.edge('memory_search', 'data_processor', label='委托')
    dot.edge('data_processor', 'qa_processor', label='调用')
    dot.edge('qa_processor', 'search_processor', label='调用')

    # 模板层
    dot.edge('memory_search', 'answer_prompt', label='加载')
    dot.edge('memory_search', 'graph_prompt', label='加载(图模式)')
    dot.edge('qa_processor', 'jinja2', label='使用')
    dot.edge('answer_prompt', 'jinja2')
    dot.edge('graph_prompt', 'jinja2')

    # 记忆层
    dot.edge('search_processor', 'mem0_client', label='调用')
    dot.edge('mem0_client', 'search_api', label='请求')
    dot.edge('mem0_client', 'graph_search', label='请求(图模式)')
    dot.edge('search_api', 'mem0_cloud')
    dot.edge('graph_search', 'mem0_cloud')

    # LLM 层
    dot.edge('qa_processor', 'openai_client', label='调用')
    dot.edge('openai_client', 'chat_api', label='请求')
    dot.edge('chat_api', 'openai_service')

    # 输出
    dot.edge('qa_processor', 'json_output', label='保存')
    dot.edge('memory_search', 'json_output', label='最终保存')

    try:
        output_path = dot.render(output_file, format=format, cleanup=True)
        print(f"✅ 架构图已生成: {output_path}")
    except Exception as e:
        print(f"❌ 生成架构图失败: {e}")

    return dot


def generate_mermaid_diagram():
    """
    生成 Mermaid 格式的流程图代码
    """
    mermaid_code = '''```mermaid
flowchart TB
    Start([开始]) --> LoadEnv[加载环境变量]
    LoadEnv --> InitClients[初始化客户端]
    InitClients --> LoadData[加载 JSON 数据]
    
    LoadData --> ConvLoop{遍历对话}
    ConvLoop --> GetSpeaker[获取说话者信息]
    GetSpeaker --> CreateIDs[创建用户 ID]
    CreateIDs --> QALoop{遍历问题}
    
    QALoop --> ExtractQ[提取问题信息]
    ExtractQ --> SearchParallel{并行搜索}
    
    SearchParallel --> Search1[搜索 Speaker 1]
    SearchParallel --> Search2[搜索 Speaker 2]
    
    Search1 --> CheckMode{图模式?}
    Search2 --> CheckMode
    
    CheckMode -->|是| GraphSearch[图记忆搜索]
    CheckMode -->|否| SemanticSearch[语义搜索]
    
    GraphSearch --> APICall[Mem0 API 调用]
    SemanticSearch --> APICall
    
    APICall --> CheckSuccess{成功?}
    CheckSuccess -->|是| ExtractMem[提取记忆]
    CheckSuccess -->|否| Retry[等待重试]
    Retry --> APICall
    
    ExtractMem --> ExtractGraph{提取图关系?}
    ExtractGraph -->|是| FormatGraph[格式化图记忆]
    ExtractGraph -->|否| FormatRes[格式化结果]
    FormatGraph --> FormatRes
    
    FormatRes --> BuildPrompt[构建提示词]
    BuildPrompt --> RenderTemplate[Jinja2 渲染]
    RenderTemplate --> OpenAICall[OpenAI API]
    OpenAICall --> Package[打包结果]
    Package --> Save[保存到文件]
    
    Save --> MoreQA{更多问题?}
    MoreQA -->|是| ExtractQ
    MoreQA -->|否| MoreConv{更多对话?}
    MoreConv -->|是| GetSpeaker
    MoreConv -->|否| Exit([完成])
    
    style Start fill:#90EE90
    style Exit fill:#90EE90
    style Search1 fill:#98FB98
    style Search2 fill:#98FB98
    style OpenAICall fill:#FFB6C1
    style Save fill:#98FB98
```
'''

    # 保存 Mermaid 代码到文件
    with open(' mermaid_search_diagram.md', 'w', encoding='utf-8') as f:
        f.write('# MemorySearch Mermaid 流程图\n\n')
        f.write(mermaid_code)
        f.write('\n\n## 如何使用\n')
        f.write('1. 将以上代码复制到支持 Mermaid 的编辑器中（如 GitHub、Typora 等）')
        f.write('\n2. 或访问 https://mermaid.live/ 在线渲染')
        f.write('\n3. 可以导出为 PNG、SVG 等格式')

    print("✅ Mermaid 流程图代码已生成: mermaid_search_diagram.md")

    return mermaid_code


def generate_mermaid_sequence_diagram():
    """
    生成时序图 Mermaid 代码
    """
    mermaid_code = '''```mermaid
sequenceDiagram
    participant User as 用户
    participant MS as MemorySearch
    participant J2 as Jinja2
    participant M0 as Mem0 API
    participant OAI as OpenAI API
    participant FS as 文件系统
    
    User->>MS: 1. 初始化
    MS->>FS: 2. 读取 JSON 文件
    FS-->>MS: 3. 返回数据
    
    par 并行搜索
        MS->>M0: 4. 搜索 Speaker 1 记忆
        M0-->>MS: 6. 返回语义记忆
        M0-->>MS: 8. 返回图关系 (图模式)
    and
        MS->>M0: 5. 搜索 Speaker 2 记忆
        M0-->>MS: 7. 返回语义记忆
        M0-->>MS: 9. 返回图关系 (图模式)
    end
    
    MS->>J2: 10. 渲染提示词模板
    J2-->>MS: 11. 返回格式化提示词
    
    MS->>OAI: 12. 发送请求
    OAI-->>MS: 13. 返回答案
    
    MS->>FS: 14. 保存结果
    MS-->>User: 15. 返回处理结果
```
'''

    with open('mermaid_search_sequence.md', 'w', encoding='utf-8') as f:
        f.write('# MemorySearch 时序图 (Mermaid)\n\n')
        f.write(mermaid_code)
        f.write('\n\n## 如何使用\n')
        f.write('访问 https://mermaid.live/ 在线渲染')

    print("✅ Mermaid 时序图代码已生成: mermaid_search_sequence.md")

    return mermaid_code


if __name__ == '__main__':
    print("=" * 60)
    print("MemorySearch 工作流流程图生成器")
    print("=" * 60)
    print()

    # 生成主工作流程图
    print("\n📊 生成主工作流程图...")
    generate_workflow_diagram()

    # 生成详细流程图
    print("\n📊 生成详细流程图...")
    generate_detailed_flow_diagram()

    # 生成时序图
    print("\n📊 生成交互时序图...")
    generate_sequence_diagram()

    # 生成架构图
    print("\n📊 生成系统架构图...")
    generate_architecture_diagram()

    # 生成 Mermaid 代码
    print("\n📊 生成 Mermaid 代码...")
    generate_mermaid_diagram()
    generate_mermaid_sequence_diagram()

    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)
    print()
    print("📁 生成的文件:")
    print("   - search_workflow_diagram.png     主工作流程图")
    print("   - search_detailed_diagram.png     详细流程图")
    print("   - search_sequence_diagram.png     交互时序图")
    print("   - search_architecture_diagram.png 系统架构图")
    print("   - mermaid_search_diagram.md       Mermaid 流程图")
    print("   - mermaid_search_sequence.md      Mermaid 时序图")
    print()
    print("💡 提示:")
    print("   - 如果生成失败，请确保已安装 graphviz")
    print("   - 在线 Mermaid 渲染: https://mermaid.live/")
    print("=" * 60)
