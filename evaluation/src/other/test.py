from openai import OpenAI
import os

client = OpenAI(base_url="http://127.0.0.1:8008/v1",api_key="vllm")
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "Hello!"}],
    temperature=0.0
)
print(response.choices[0].message.content)
# from langgraph.prebuilt import create_react_agent
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from transformers import BitsAndBytesConfig
# # from langchain_ollama import ChatOllama

# # llm = ChatOllama(
# #     model="llama3.1",
# #     temperature=0,
# #     # other params...
# # )
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
# )
# llm = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen2.5-1.5B-Instruct",
#     task="text-generation",
#     device_map="auto",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
#     model_kwargs={"quantization_config": quantization_config}, 
# )
# model = ChatHuggingFace(llm=llm)

# def tool() -> None:
#     """Testing tool."""
#     print("This is a test tool.")

# agent = create_react_agent(
#     model=model,
#     tools=[tool],
# )
# # print(f"--------------{dir(agent)}")
# response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
# print(response["messages"][-1].content)
