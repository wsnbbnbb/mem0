import os
import sys
# print(sys.path)
import json

from mem0 import Memory
os.environ["OPENAI_API_KEY"] = "your-api-key" # for embedder

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            # "model": "qwen2.5:7b",
             "model": "gpt-oss:20b",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
    }
}
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
    # "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    # },
}
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""  
from tqdm import tqdm
          
def getData():
    import json
    with open("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo10.json", "r") as f:
        data = json.load(f)

    item = data[0]
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    speaker_a_user_id = f"{speaker_a}_{0}"
    speaker_b_user_id = f"{speaker_b}_{0}"

    key  = 'session_1'            
    date_time_key = key + "_date_time"
    timestamp = conversation[date_time_key]
    chats = conversation[key]

    messages = []
    messages_reverse = []
    for chat in chats:
                    if chat["speaker"] == speaker_a:
                        messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                        messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                    elif chat["speaker"] == speaker_b:
                        messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                        messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                    else:
                        raise ValueError(f"Unknown speaker: {chat['speaker']}")
    return messages, speaker_a_user_id, timestamp
def main():
    m = Memory.from_config(config)
    import json
    with open("/root/nfs/hmj/proj/mem0/evaluation/src/dataset/locomo10.json", "r") as f:
        data = json.load(f)

    item = data[0]
    conversation = item["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    speaker_a_user_id = f"{speaker_a}_{0}"
    speaker_b_user_id = f"{speaker_b}_{0}"

    key  = 'session_19'            
    date_time_key = key + "_date_time"
    timestamp = conversation[date_time_key]
    chats = conversation[key]

    messages = []
    messages_reverse = []
    for chat in chats:
                    if chat["speaker"] == speaker_a:
                        messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                        messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                    elif chat["speaker"] == speaker_b:
                        messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                        messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                    else:
                        raise ValueError(f"Unknown speaker: {chat['speaker']}")
    m.add(messages, user_id=speaker_a_user_id, metadata={"timestamp": timestamp},prompt=custom_instructions)

    result = m.search(user_id=speaker_a_user_id, query="What are some memorable experiences Caroline has had with her family during camping trips?")
    # result_pro = result['results']
    # for res in result_pro:
    #     print(res['memory'])
    print(f"----------\nSearch Results:\n {result}")
         
import inspect

def count_class_methods_lines(cls, ignore_comments=True):
    result = {}
    
    for name, member in cls.__dict__.items():
        func = None
        if inspect.isfunction(member):
            func = member
        elif isinstance(member, (staticmethod, classmethod)):
            func = member.__func__

        if func:
            try:
                source = inspect.getsource(func)
                lines = source.splitlines()
                if ignore_comments:
                    lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                result[name] = len(lines)
            except OSError:
                result[name] = None

    return result
def testAdd():
    
    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "qwen3:8b",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
        }
    }

    m = Memory.from_config(config)
    messages = [
        {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
        {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
        {"role": "user", "content": "I'm not a big fan of thriller movies but I love sci-fi movies."},
        {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
    ]
    
    m.add(messages, user_id="alice", metadata={"category": "movies"})
  
    rs = m.search("did user see a film", user_id="alice")
    print(f"Search Results: {rs}")
    
    
def graphAdd():
   
    config = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "vllm_base_url": "http://localhost:8000/v1",
            "api_key": "vllm-api-key",
            "temperature": 0, # temperature 0.7表示生成内容有一定的随机性，适合创意写作等场景；temperature 0表示生成内容更确定，适合需要精确回答的场景。
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
    # config = {
    #     "llm": {
    #     "provider": "vllm",
    #     "config": {
    #         "model": "Qwen/Qwen2.5-7B-Instruct",
    #         "vllm_base_url": "http://localhost:8000/v1",
    #         "api_key": "vllm-api-key",
    #         "temperature": 0.7,
    #         "max_tokens": 2000,
    #     },
    # },
    # "embedder": {"provider": "huggingface", "config": {"model": "all-MiniLM-L6-v2"}},
    # "vector_store": {
    #     "provider": "qdrant",
    #     "config": {"collection_name": "vllm_memories", "host": "localhost", "port": 6333},
    # },
    #     "graph_store": {
    #         "provider": "neo4j",# or neo4j-community
    #         "config": {
    #             "url": "bolt://localhost:7687",
    #             "username": "neo4j", # or neo4j
    #             "password": "Neo4j2025",
    #             "database": "neo4j",
    #         }
    #     }
    # }

    memory = Memory.from_config(config)

  
    messages, user_id, timestamp = getData()
    batch_size = 2
    for i in tqdm(range(0, len(messages),batch_size)):
            batch_messages = messages[i : i + batch_size]
            memory.add( batch_messages, user_id=user_id,metadata={"timestamp": timestamp},prompt=custom_instructions)

    # memory.add(messages, user_id=user_id)
    results = memory.search(
        "What does Caroline love most about camping with her family?",
        user_id=user_id,
        limit=3,
        # rerank=True,
    )

    for hit in results["results"]:
        print(hit["memory"])
def printDict():
    a = {'type': 'function', 'function': {'name': 'establish_relationships', 'description': 'Establish relationships among the entities based on the provided text.', 'parameters': {'type': 'object', 'properties': {'entities': {'type': 'array', 'items': {'type': 'object', 'properties': {'source': {'type': 'string', 'description': 'The source entity of the relationship.'}, 'relationship': {'type': 'string', 'description': 'The relationship between the source and destination entities.'}, 'destination': {'type': 'string', 'description': 'The destination entity of the relationship.'}}, 'required': ['source', 'relationship', 'destination'], 'additionalProperties': False}}}, 'required': ['entities'], 'additionalProperties': False}}}
    # 将a 格式化输出
    b = {'content': '', 'tool_calls': [{'name': 'establish_relationships', 'arguments': {'entities': ['caroline', 'mel', 'melanie', 'kids', 'work']}}]}
    
    print(json.dumps(b, indent=2))
if __name__ == "__main__":
    main()
    # graphAdd()
    # printDict()
    # testAdd()
    # from mem0.memory.graph_memory import MemoryGraph
    # print(count_class_methods_lines(MemoryGraph))
    # print("-----")
    # print(count_class_methods_lines(Memory))