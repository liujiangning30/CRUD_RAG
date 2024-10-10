import os
import openai
import random
from loguru import logger
from typing import List, Union

from src.llms.base import BaseLLM
from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")

class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: str) -> str:
        openai.api_key = conf.GPT_api_key
        if conf.GPT_api_base and conf.GPT_api_base.strip():
            openai.base_url = conf.GPT_api_base
        res = openai.chat.completions.create(
            model = self.params['model_name'],
            messages = [{"role": "user","content": query}],
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
        )
        real_res = res.choices[0].message.content

        token_consumed = res.usage.total_tokens
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res


class GPTBatched(BaseLLM):
    def __init__(self,
                 model_name='gpt-4-turbo',
                 key=os.environ.get('OPENAI_API_KEY', 'YOUR OPENAI API KEY'),
                 proxies=dict(
                    http='http://liujiangning:QvNIdAiv3QkiXOB3Kpx24kum6KpEievWYfbu1cPO0FJRqDPU8Zo1nz79bolY@closeai-proxy.pjlab.org.cn:23128',
                    https='http://liujiangning:QvNIdAiv3QkiXOB3Kpx24kum6KpEievWYfbu1cPO0FJRqDPU8Zo1nz79bolY@closeai-proxy.pjlab.org.cn:23128'
                 ),
                 max_new_tokens: int = 512,
                 top_p: float = 0.8,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 report=False):
        from lagent.llms import GPTAPI
        llm = GPTAPI(
            model_type=model_name,
            key=key,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            proxies=proxies,
            query_per_second=50,
            retry=1000)
        self.llm = llm

        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: Union[str, List[str]], **gen_params) -> str:
        if isinstance(query, str):
            query = [query]
        query = [[dict(role='user', content=q)] for q in query]
        real_res = self.llm.chat(query, **gen_params)
        return real_res


class InternLMClient(BaseLLM):
    def __init__(self,
                 model_name='internlm2_5-7b-chat',
                 url='http://127.0.0.1:23333',
                 max_new_tokens: int = 512,
                 top_p: float = 0.8,
                 top_k: float = 40,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 stop_words: List[str] = ['<|im_end|>'],
                 report=False):

        from lagent.llms import INTERNLM2_META, LMDeployClient
        llm = LMDeployClient(
            model_name=model_name,
            url=url,
            meta_template=INTERNLM2_META,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words)
        self.llm = llm

        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

    def request(self, query: Union[str, List[str]], **gen_params) -> str:
        if isinstance(query, str):
            query = [query]
        query_pattern = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        query = [query_pattern.format(query=q) for q in query]
        real_res = self.llm.generate(query, **gen_params)
        return real_res
