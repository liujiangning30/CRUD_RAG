import json
import os
import random
from typing import Any

from src.datasets.base import BaseDataset

class Xinhua(BaseDataset):
    def __init__(self, data, shuffle: bool = False, seed: int = 22):
        self.data = data

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: int | slice) -> dict | list[dict]:
            return self.data[key]

    def load(self) -> list[dict]:
        return self.data[:]

    def statistics(self) -> dict:
        stat = {'doc': 0, 'gen': 0, 'kno': 0, 'num': 0}
        for type_ in stat.keys():
            stat[type_] = sum([obj['type']==type_ for obj in self.data])
        return stat


def get_task_datasets(path: str, task: str, w_ctx: bool = False, shuffle: bool = False, seed: int = 22):
    if w_ctx:
        if task == "all":
            sub_tasks = ["event_summary", "continuing_writing", "hallu_modified", "questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
        elif task == "quest_answer":
            sub_tasks = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
        else:
            sub_tasks = [task]
        id2task_name = {
            "event_summary": "Summary",
            "continuing_writing": "ContinueWriting",
            "hallu_modified": "HalluModified",
            "questanswer_1doc": "QuestAnswer1Doc",
            "questanswer_2docs": "QuestAnswer2Docs",
            "questanswer_3docs": "QuestAnswer3Docs"
        }
        data = dict()
        for sub_task in sub_tasks:
            sub_task_path = path.replace('split_merged', f'split_{id2task_name[sub_task]}_w_ctx')
            with open(sub_task_path) as fr:
                data[sub_task] = json.load(fr)
    else:
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
        
    if task == "all":
        return [
            Xinhua(data["event_summary"], shuffle, seed), 
            Xinhua(data["continuing_writing"], shuffle, seed),
            Xinhua(data["hallu_modified"], shuffle, seed),
            Xinhua(data["questanswer_1doc"], shuffle, seed),
            Xinhua(data["questanswer_2docs"], shuffle, seed),
            Xinhua(data["questanswer_3docs"], shuffle, seed),
        ]
    elif task == "quest_answer":
        return [
            Xinhua(data["questanswer_1doc"], shuffle, seed),
            Xinhua(data["questanswer_2docs"], shuffle, seed),
            Xinhua(data["questanswer_3docs"], shuffle, seed),
        ]
    else:
        return [Xinhua(data[task], shuffle, seed)]
