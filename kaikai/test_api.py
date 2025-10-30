#export YUANJING_API_KEY='*****'
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("YUANJING_API_KEY"),
    base_url="https://maas-api.ai-yuanjing.com/openapi/compatible-mode/v1",
)

model_name = 'deepseek-r1'


try:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'user', 'content': '你是谁？'}
            # {"role": "user", "content": [{"type": "text", "text": "who are you?"}]}
        ]
    )
except Exception as err:
    print(f"调用模型失败：{err}")
else:
    if not getattr(completion, "choices", None):
        print("调用模型失败：未返回任何结果")
    else:
        print("调用模型成功。原始响应：")
        print("最终答案：")
        print(completion.choices[0].message.content)

