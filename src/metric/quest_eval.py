import os
import re
import json
import jieba
import requests
from tqdm import tqdm
import numpy as np
from loguru import logger
from typing import List, Union
from collections import Counter

from src.llms import GPT, InternLMClient, GPTBatched
from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")
json_response = '''
{\"key_info\": ["新增并网光伏发电容量1060万千瓦", "四分之一", "全国新增光伏电站855万千瓦", "分布式光伏容量205万千瓦", "2014年中国光伏发电量250亿千瓦。", "同比增长超过200%"], 

\"question\": ["2014年中国新增并网光伏发电容量是多少？", "2014年中国新增并网光伏发电容量约占全球新增容量的几分之几？","全国新增光伏电站的容量是多少？", "分布式光伏容量是多少？", "2014年中国光伏发电量是多少？", "2014年中国光伏发电量相比前一年增长了多少？"]}
'''

class QuestEval(GPT):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False, task_name='summary'):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report
        
        self.quest_gt_save = self._read_quest_gt(f'{task_name}_quest_gt_save.json')

    def save_quest_gt(self, task_name):
        with open(f'src/quest_eval/{task_name}_quest_gt_save.json', 'w', encoding='utf-8') as f:
            json.dump(self.quest_gt_save, f, ensure_ascii=False, indent=4)

    def question_generation(self, text4gen: Union[str, List[str]]):
        prompt = self._read_prompt_template("quest_eval_gen.txt")
        if isinstance(text4gen, str):
            text4gen = [text4gen]
        query = [prompt.format(json_response=json_response, news=item) for item in text4gen]
        extracted_content = self.safe_request(query)
        question4eval = []
        for item in extracted_content:
            try:
                json_item = json.loads(item)
            except Exception as e:
                logger.warning(str(e))
                json_item = dict(
                    key_info=[''],
                    question=['']
                )
            question4eval.append(json_item)
        return question4eval

    def question_answer(self, context: Union[str, List[str]], question: Union[str, List[str]]):
        template = self._read_prompt_template('quest_eval_answer.txt')
        if isinstance(context, str):
            context = [context]
        if isinstance(question, str):
            question = [question]
        assert len(context) == len(question)
        query = [template.format(context=item[0], questions=item[1]) for item in zip(context, question)]
        answers = self.safe_request(query)
        
        pattern = r'<response>\n(.*?)\n</response>'
        real_answers = []
        for i, answer in enumerate(answers):
            real_answers.append(re.findall(pattern, answer, re.DOTALL))
        return real_answers
    
    def _read_prompt_template(self, filename: str):
        path = os.path.join('src/prompts/', filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''
        
    def _read_quest_gt(self, filename: str) -> dict:
        path = os.path.join('src/quest_eval', filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.loads(f.read())
        else:
            logger.error(f'Questions generated from ground truth for evaluation not found at {path}')
            return {}
    
    def get_QA_pair(self, data_point: dict):
        ground_truth_text = data_point["ground_truth_text"]
        generated_text = data_point["generated_text"]
        
        if data_point["ID"] in self.quest_gt_save.keys():
            questions_gt = self.quest_gt_save[data_point["ID"]]["question"]
            answers_gt4gt = self.quest_gt_save[data_point["ID"]]["answers"]
        else:
            keyinfo_and_questions = self.question_generation(ground_truth_text)
            questions_gt = keyinfo_and_questions["question"]           
            answers_gt4gt = self.question_answer(ground_truth_text, questions_gt) # 用ground truth回答ground truth生成的问题
            
            keyinfo_and_questions["answers"] = answers_gt4gt
            self.quest_gt_save[data_point["ID"]] = keyinfo_and_questions
    
        answers_gm4gt = self.question_answer(generated_text, questions_gt) # 用generated text回答ground truth生成的问题

        return questions_gt, answers_gt4gt, answers_gm4gt

    def batch_get_QA_pair(self, dataset: List[dict], batch_size: int = 8, show_progress_bar=False, task_name=''):
        ground_truth_texts = []
        generated_texts = []
        uncached_ids = []
        cached_ids_wo_gm4gt = []

        cached_dataset = []
        uncached_dataset = []
        cached_dataset_wo_gm4gt = []
        for data_point in dataset:
            data_id = data_point["ID"]
            if data_id in self.quest_gt_save.keys():
                data_point['questions_gt'] = self.quest_gt_save[data_id]["question"]
                data_point['answers_gt4gt'] = self.quest_gt_save[data_id]["answers"]
                # ***NOTE: for test. You should comment out the following line. ***
                self.quest_gt_save[data_id].pop('answers_gm4gt', None)
                if 'answers_gm4gt' not in self.quest_gt_save[data_id]:
                    cached_dataset_wo_gm4gt.append(data_point)
                    cached_ids_wo_gm4gt.append(data_id)
                    continue
                data_point['answers_gm4gt'] = self.quest_gt_save[data_id]["answers_gm4gt"]
                cached_dataset.append(data_point)
            else:
                uncached_dataset.append(data_point)
                ground_truth_texts.append(data_point["ground_truth_text"])
                generated_texts.append(data_point["generated_text"])
                uncached_ids.append(data_id)
        del dataset

        if cached_dataset_wo_gm4gt:
            all_generated_answers_gm4gt = []
            spilt_ids = range(0, len(cached_dataset_wo_gm4gt), batch_size)
            for start in (tqdm(spilt_ids, desc=f"quest eval for cached task {task_name}") if show_progress_bar else spilt_ids):
                end = start + batch_size
                generated_text_batch = [data_point["generated_text"] for data_point in cached_dataset_wo_gm4gt[start: end]]
                questions_gt_batch = [data_point["questions_gt"] for data_point in cached_dataset_wo_gm4gt[start: end]]
                generated_answers_gm4gt = self.question_answer(generated_text_batch, questions_gt_batch)
                all_generated_answers_gm4gt.extend(generated_answers_gm4gt)
            for i, id_ in enumerate(cached_ids_wo_gm4gt):
                answers_gm4gt = all_generated_answers_gm4gt[i]
                self.quest_gt_save[id_]["answers_gm4gt"] = answers_gm4gt
                # 保存结果
                cached_dataset_wo_gm4gt[i]['answers_gm4gt'] = answers_gm4gt

        if uncached_dataset:
            all_generated_questions = []
            all_generated_answers_gt4gt = []
            all_generated_answers_gm4gt = []
            spilt_ids = range(0, len(uncached_dataset), batch_size)
            for start in (tqdm(spilt_ids, desc=f"quest eval for uncached task {task_name}") if show_progress_bar else spilt_ids):
                end = start + batch_size
                ground_truth_batch = ground_truth_texts[start:end]
                generated_text_batch = generated_texts[start:end]
                generated_questions = self.question_generation(ground_truth_batch)
                generated_answers_gt4gt = self.question_answer(ground_truth_batch, [gq["question"] for gq in generated_questions]) 
                generated_answers_gm4gt = self.question_answer(generated_text_batch, [gq["question"] for gq in generated_questions])
                all_generated_questions.extend(generated_questions)
                all_generated_answers_gt4gt.extend(generated_answers_gt4gt)
                all_generated_answers_gm4gt.extend(generated_answers_gm4gt)
            for i, id_ in enumerate(uncached_ids):
                keyinfo_and_questions = all_generated_questions[i]
                answers_gt4gt = all_generated_answers_gt4gt[i]
                answers_gm4gt = all_generated_answers_gm4gt[i]
                questions_gt = keyinfo_and_questions["question"]
                keyinfo_and_questions["answers"] = answers_gt4gt
                keyinfo_and_questions["answers_gm4gt"] = answers_gm4gt
                # 缓存生成的结果
                self.quest_gt_save[id_] = keyinfo_and_questions
                # 保存结果
                uncached_dataset[i]['questions_gt'] = questions_gt
                uncached_dataset[i]['answers_gt4gt'] = answers_gt4gt
                uncached_dataset[i]['answers_gm4gt'] = answers_gm4gt
        return cached_dataset + cached_dataset_wo_gm4gt + uncached_dataset

    def quest_eval(self, data_point: dict):
        try:
            if "questions_gt" in data_point and "answers_gt4gt" in data_point and "answers_gm4gt" in data_point:
                questions_gt = data_point.pop("questions_gt")
                answers_gt4gt = data_point.pop("answers_gt4gt")
                answers_gm4gt = data_point.pop("answers_gm4gt")
            else:
                questions_gt, answers_gt4gt, answers_gm4gt = self.get_QA_pair(data_point)

            quest_eval_save = {}
            min_size = min([len(questions_gt), len(answers_gt4gt)])
            questions_gt = questions_gt[:min_size]
            answers_gt4gt = answers_gt4gt[:min_size]
            if len(answers_gm4gt) > min_size:
                answers_gm4gt = answers_gm4gt[:min_size]
            else:
                answers_gm4gt.extend(['无法推断'] * (min_size - len(answers_gm4gt)))
            quest_eval_save["questions_gt"] = questions_gt
            quest_eval_save["answers_gt4gt"] = answers_gt4gt
            quest_eval_save["answers_gm4gt"] = answers_gm4gt

            # 去除ground truth无法推断的问题，说明生成的问题不好，需要排除
            indices = [i for i, x in enumerate(answers_gt4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]

            if len(answers_gm4gt) == 0:
                return 0, 0, quest_eval_save

            undetermined_ratio = answers_gm4gt.count("无法推断") / len(answers_gm4gt)
            quest_recall = 1 - undetermined_ratio

            indices = [i for i, x in enumerate(answers_gm4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]
            
            if answers_gm4gt == []:
                return 0, 0, quest_eval_save

            quest_avg_f1 = word_based_f1_score(answers_gt4gt, answers_gm4gt)

        except Exception as e:
            logger.warning(repr(e))
            quest_eval_save = {}
            quest_eval_save["questions_gt"] = []
            quest_eval_save["answers_gt4gt"] = []
            quest_eval_save["answers_gm4gt"] = []
            return 0, 0, quest_eval_save
        
        return quest_avg_f1, quest_recall, quest_eval_save

    
class QuestEvalGPTBatched(QuestEval):
    """lagent based
    """
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
                 report=False,
                 task_name='summary'):
        llm = GPTBatched(
            model_name=model_name,
            key=key,
            proxies=proxies,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty)
        self.llm = llm
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            report=report,
            task_name=task_name)
        self.report = report
        self.quest_gt_save = self._read_quest_gt(f'{task_name}_quest_gt_save.json')

    def request(self, query: str) -> str:
        responses = self.llm.request(query)
        return responses


class QuestEvalAPI(QuestEval):
    """lagent based
    """
    def __init__(self,
                 model_name='Meta-Llama-3-8B-Instruct',
                 url='http://127.0.0.1:23333',
                 temperature=1.0,
                 max_new_tokens=1024,
                 report=False,
                 task_name='summary'):
        llm = InternLMClient(
            model_name=model_name,
            url=url,
            temperature=temperature,
            max_new_tokens=max_new_tokens)
        self.llm = llm
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            report=report,
            task_name=task_name)
        self.report = report
        self.quest_gt_save = self._read_quest_gt(f'{task_name}_quest_gt_save.json')

    def request(self, query: str) -> str:
        responses = self.llm.request(query)
        return responses


def compute_f1(a_gold, a_pred):
    gold_toks = list(jieba.cut(a_gold)) 
    pred_toks = list(jieba.cut(a_pred)) 
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def word_based_f1_score(a_gold_list, a_pred_list):
    f1_list=[]
    for a_gold,a_pred in zip(a_gold_list, a_pred_list):
        f1_list.append(compute_f1(a_gold,a_pred))
    return np.mean(f1_list)

