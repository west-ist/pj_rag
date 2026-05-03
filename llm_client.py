#调用LLM
from dashscope import Generation
from http import HTTPStatus
import config
def call_llm(prompt:str, system_message:str = "你是一个专业的法律助手，请基于提供的上下文准确回答问题。")->str:
    error = "调用失败"
    if not config.Dashscope_key or config.Dashscope_key == "":
        print("API密钥未配置\n")
        return error

    try:
        response = Generation.call(
            model=config.model,
            prompt=prompt,
            system=system_message,
            api_key=config.Dashscope_key,
            result_format='message',
            temperature=0.1,
            seed=42  #固定随机种子
        )
        if response.status_code != HTTPStatus.OK:
            return error
        return response.output.choices[0].message.content
    except Exception:
        return error

