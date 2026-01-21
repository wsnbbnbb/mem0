import time
from openai import OpenAI
import subprocess
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig


# -------------------------
# vLLM Test
# -------------------------
def test_vllm():
    print("\n=== Testing vLLM ===")
    client = OpenAI(base_url="http://127.0.0.1:8008/v1", api_key="vllm")

    start = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.0
    )
    end = time.time()

    print("Output:", response.choices[0].message.content)
    print("Time:", end - start, "sec")
    return end - start


# -------------------------
# Ollama Test
# -------------------------
def test_ollama():
    print("\n=== Testing Ollama ===")

    # 一般用 subprocess 调用 ollama run
    start = time.time()
    result = subprocess.check_output(
        ["ollama", "run", "qwen2.5:7b"],
        input="Hello!",
        text=True
    )
    end = time.time()

    print("Output:", result.strip())
    print("Time:", end - start, "sec")
    return end - start


# -------------------------
# HuggingFace Transformers Test (bnb 4bit)
# -------------------------
def test_huggingface():
    print("\n=== Testing HuggingFace Transformers (4-bit) ===")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        device_map="cuda:4",
        pipeline_kwargs=dict(
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.03,
        ),
        model_kwargs={"quantization_config": quant_config},
    )

    model = ChatHuggingFace(llm=llm)

    start = time.time()
    result = model.invoke("Hello!")
    end = time.time()

    print("Output:", result.content)
    print("Time:", end - start, "sec")
    return end - start


# -------------------------
# RUN ALL
# -------------------------
def run_all_tests():
    times = {}

    try:
        times["vLLM"] = test_vllm()
    except Exception as e:
        print("vLLM Error:", e)

    try:
        times["Ollama"] = test_ollama()
    except Exception as e:
        print("Ollama Error:", e)

    try:
        times["HuggingFace"] = test_huggingface()
    except Exception as e:
        print("HuggingFace Error:", e)

    print("\n=== Summary ===")
    for k, v in times.items():
        print(f"{k}: {v:.4f} sec")

def fname():
    # pip install -qU "langchain[anthropic]" to call the model

    from langgraph.prebuilt import create_react_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        prompt="You are a helpful assistant"
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # # run_all_tests()
    # test_huggingface()
    # test_ollama()
    fname()
