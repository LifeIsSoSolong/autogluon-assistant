import os
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  
    base_url="https://oa.api2d.net/v1",
)

model_name = 'o3-mini'
# model_name = 'gpt-5'

try:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            # {'role': 'user', 'content': '你是谁？'}
            {"role": "user", "content": [{"type": "text", "text": "who are you?"}]}
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
