import os
import json
import pandas as pd

def convert_to_table(results, task_name):
    metrics_mapping = {
        "avg. bleu-avg": "bleu",
        "avg. rouge-L": "rouge-L",
        "QA_avg_F1": "RAGQuestEval-precision",
        "QA_recall": "RAGQuestEval-recall",
        "avg. length": "length"
    }
    ordered_metrics = ['bleu', 'rouge-L', 'RAGQuestEval-recall', 'RAGQuestEval-precision', 'length']
    data = dict()
    for key, value in results['overall'].items():
        if key in metrics_mapping:
            metric_name = metrics_mapping[key]
            value_percent = value * 100 if not key.endswith('length') else value
            data[metric_name] = [task_name, metric_name, "%.2f" % value_percent]
    new_results = [data[metric_name] for metric_name in ordered_metrics]
    return new_results


if __name__ == "__main__":
    overall_results = []
    id2task_name = {
        "continuing_writing": "ContinueWriting",
        "hallu_modified": "HalluModified",
        "questanswer_1doc": "QuestAnswer1Doc",
        "questanswer_2docs": "QuestAnswer2Docs",
        "questanswer_3docs": "QuestAnswer3Docs",
        "event_summary": "Summary",
    }
    for task_name in id2task_name.values():
        task_result_file = os.path.join('output/docs_80k_chuncksize_128_0_top8_use_gt_ctx1_inject_negative_ctx1_InternLMClient', f'{task_name}_internlm2_5-7b-chat_quest_eval_gpt-4o-mini.json')
        if not os.path.exists(task_result_file):
            continue
        with open(task_result_file, 'r') as fr:
            task_result = json.load(fr)
        task_metrics = convert_to_table(task_result, task_name)
        overall_results.extend(task_metrics)
    df = pd.DataFrame(overall_results, columns=['task', 'metric', 'value'])
    df.to_csv('output/docs_80k_chuncksize_128_0_top8_use_gt_ctx1_inject_negative_ctx1_InternLMClient/overall_results_internlm2_5-7b-chat_quest_eval_gpt-4o-mini.csv', index=False)