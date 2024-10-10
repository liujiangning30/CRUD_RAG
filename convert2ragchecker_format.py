import os
import json
import codecs
import argparse


def convert_data(task, model_name, prediction_dir, output_file):
    prediction_dir = os.path.join(prediction_dir, 'temp', f'{task}_{model_name}')
    predictions = []
    for file_name in os.listdir(prediction_dir):
        prediction_file = os.path.join(prediction_dir, file_name)
        with codecs.open(prediction_file, 'r', 'utf-8') as fr:
            prediction = json.load(fr)
            predictions.append(prediction)
    if task == 'Summary':
        query_key = 'event'
        gt_key = 'summary'
    elif task == 'ContinueWriting':
        query_key = 'beginning'
        gt_key = 'continuing'
    elif task == 'HalluModified':
        query_key = 'newsBeginning'
        gt_key = 'hallucinatedMod'
    elif task.startswith('QuestAnswer'):
        query_key = 'questions'
        gt_key = 'answers'
    results = []
    for prediction in predictions:
        query_id = prediction["ID"]
        query = prediction[query_key]
        gt_answer = prediction[gt_key]
        retrieved_context = [dict(doc_id=str(i), text=p) for i, p in enumerate(prediction['retrieve_context'].split('\n\n'))]
        response = prediction['generated_text']

        result = {
            "query_id": query_id,
            "query": query if task.startswith('QuestAnswer') else None,
            "gt_answer": gt_answer,
            "retrieved_context": retrieved_context,
            "response": response
        }
        results.append(result)

    output_dir = os.path.split(output_file)[0]
    os.makedirs(output_dir, exist_ok=True)
    with codecs.open(output_file, 'w', 'utf-8') as fw:
        json.dump(dict(results=results), fw, ensure_ascii=False)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, help="Name of task.")
    parser.add_argument('--model_name', type=str, help="Name of the test model.")
    parser.add_argument('--prediction_dir', type=str, help="Dir of prediction data being converted.")
    parser.add_argument('--output_file', type=str, help="File of converted data to be saved.")
    args = parser.parse_args()

    convert_data(task=args.task, model_name=args.model_name, prediction_dir=args.prediction_dir, output_file=args.output_file)
