
import sys
sys.path.append("/root/nfs/hmj/proj/mem0")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
_tools = [EXTRACT_ENTITIES_TOOL]
import json
from mem0.llms.ollama import OllamaLLM
from mem0.configs.llms.base import BaseLlmConfig
# config_chat = BaseLlmConfig(
#             model='gpt-oss:20b',
#             temperature=0,
#             max_tokens=8192,
#             top_p=1.0
#         )
from mem0 import Memory
config_g = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
            "api_key": "vllm-api-key",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    },
     "graph_store": {
            "provider": "neo4j",# or neo4j-community
            "config": {
                "url": "bolt://localhost:7687",
                "username": "neo4j", # or neo4j
                "password": "Neo4j2025",
                "database": "neo4j",
            },
        },
}
import re
def main():
    llm = Memory.from_config(config_g).llm
    filters = {
        "user_id": "Melanie",
    }
    data = "Caroline: Hey Mel! Good to see you! How have you been?\n\
    Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?"
    search_results = llm.generate_response(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                    },
                    {"role": "user", "content": data},
                ],
                tools=_tools,
                tool_choice="none",
            )
    tool_calls = search_results.get("tool_calls", [])
    entity_type_map = {}
    if not tool_calls or len(tool_calls) == 0:
            # 2. 如果 tool_calls 为空，尝试从 content 里提取 <tool_call>
            content = search_results.get("content", "")
            match = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", content, re.S)
            if match:
                tool_call_json = json.loads(match.group(1))
                tool_calls = [tool_call_json]
        # try:
        # 3. 遍历 tool_calls 提取实体
    for tool_call in tool_calls:
                if tool_call.get("name") != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    entity_type_map[item["entity"]] = item["entity_type"]
    print(entity_type_map)      
# search_results 已经是 dict
    # print(json.dumps(search_results, indent=4, ensure_ascii=False))
# def print_tools():
def print_tools():
    llm = Memory.from_config(config_g).llm
    _tools= [RELATIONS_TOOL]
    messages =[
        {
        'role': 'system', 
         'content': '\n\nYou are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:\n\n1. Extract only explicitly stated information from the text.\n2. Establish relationships among the entities provided.\n3. Use "user_id: Caroline_0" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.\nCUSTOM_PROMPT\n\nRelationships:\n    - Use consistent, general, and timeless relationship types.\n    - Example: Prefer "professor" over "became_professor."\n    - Relationships should only be established among the entities explicitly mentioned in the user message.\n\nEntity Consistency:\n    - Ensure that relationships are coherent and logically align with the context of the message.\n    - Maintain consistent naming for entities across the extracted data.\n\nStrive to construct a coherent and easily understandable knowledge graph by eshtablishing all the relationships among the entities and adherence to the user’s context.\n\nAdhere strictly to these guidelines to ensure high-quality knowledge graph extraction.'}, {'role': 'user', 'content': "List of entities: ['caroline', 'melanie', 'lgbtq_support_group', 'yesterday', 'caroline_0']. \n\nText: Caroline: I went to a LGBTQ support group yesterday and it was so powerful.\nMelanie: Wow, that's cool, Caroline! What happened that was so awesome? Did you hear any inspiring stories?"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
        ]

    extracted_entities = llm.generate_response(
            messages=messages,
            tools=_tools,
            tool_choice="none",# 设置为none，强制使用工具调用
        #      tool_choice={
        # "type": "function",
        # "function": {"name": "establish_relations"}   # <= 强制工具调用
    # }
        )
    
    print(json.dumps(extracted_entities, indent=4, ensure_ascii=False))
    # print(json.dumps(_tools, indent=4, ensure_ascii=False))
def print1():
    search_results = {'content': '<tool_call>\n{"name": "extract_entities", "arguments": {"entities": [{"entity": "Caroline", "entity_type": "Person"}, {"entity": "Melanie", "entity_type": "Person"}]}}\n</tool_call>', 'tool_calls': []}
    entity_type_map = {}
    try:
            for tool_call in search_results["tool_calls"]:
                if tool_call["name"] != "extract_entities":
                    continue
                for item in tool_call["arguments"]["entities"]:
                    # logger.info(f"提取实体item:\n{item.keys()}:{item.values()}")
                    entity_type_map[item["entity"]] = item["entity_type"]
                    # entity_type_map[item["name"]] = item["type"]
    except Exception as e:
            # logger.exception(
            #     f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            # )
            print(f"Error in search tool: {e}, search_results={search_results}")

    entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
    print(entity_type_map)
if __name__ == "__main__":
    main()
    # print1()
    # print_tools()