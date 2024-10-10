from lmdeploy.serve.openai.api_client import APIClient

# model_path = '/cpfs01/shared/public/llmeval/model_weights/hf_hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa'
client = APIClient("http://22.8.32.69:23333")
inputs = [
    '你好',
    '今天天气怎么样',
    '你是谁',
    '帮我写一首以梅花为主题的五言律诗',
    '5+2等于多少',
    '中国四大发明'
]
for text in client.completions_v1(
        'internlm2_5-7b-chat',
        inputs):
    pass
print(text)

# from lmdeploy import pipeline

# pipe = pipeline(model_path)
# response = pipe(inputs)
# for resp in response:
#     print(resp.text or resp['text'])

