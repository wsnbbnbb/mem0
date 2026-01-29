import os
from typing import Dict, List, Optional

from openai import OpenAI
import sys
sys.path.append("/root/nfs/hmj/proj/mem0/evaluation")
from AgentMem.configs.llms.base import BaseLlmConfig
from AgentMem.llms.base import LLMBase


class OpenAIStructuredLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "gpt-4o-2024-08-06"

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.openai_base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> str:
        """
        Generate a response based on the given messages using OpenAI.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries, each containing a 'role' and 'content' key.
            response_format (Optional[str]): The desired format of the response. Defaults to None.


        Returns:
            str: The generated response.
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.beta.chat.completions.parse(**params)
        return response.choices[0].message.content


def test():
   # LLM 配置
    config_data = {
        "model": "arcee-ai/trinity-large-preview:free",
        # 修改：移除末尾的 "/chat/completions"，只保留基础根路径
        "openai_base_url": "https://openrouter.ai/api/v1",
        "api_key": "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",
        "temperature": 0,
        "max_tokens": 2000,
    }
    
    # 将字典转换为 BaseLlmConfig 对象
    config_obj = BaseLlmConfig(**config_data)
    
    # 传入对象而不是字典
    llm=OpenAIStructuredLLM(config=config_obj)
    
    resp = llm.generate_response(messages=[{"role": "user", "content": "Hello, world!"}])
    print(resp)
if __name__ == "__main__":
    test()  