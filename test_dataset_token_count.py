import os
import json
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

def tokens_counter(text: str):
    tokens = encoding.encode(text)
    token_count = len(tokens)
    return token_count

response_pattern = """
{query}
<response>
{answer}
</response>
"""


def run(task_name):
    with open(f'output/docs_80k_chuncksize_128_0_top8_InternLMClient/{task_name}_internlm2_5-7b-chat.json', 'r', encoding='utf-8') as fr:
        results = json.load(fr)
    with open(os.path.join('src/prompts/', 'quest_eval_answer.txt'), 'r', encoding='utf-8') as fr:
        template = fr.read()

    num_input_tokens = 0
    num_output_tokens = 0
    for data_point in results['results']:
        generated_text = data_point['log']['generated_text']
        questions_gt = data_point['log']['quest_eval_save']['questions_gt']
        answers_gt4gt = data_point['log']['quest_eval_save']['answers_gt4gt']
        input_text = template.format(context=generated_text, questions=questions_gt)
        output_text = '\n'.join([response_pattern.format(query=q, answer=a) for q, a in zip(questions_gt, answers_gt4gt)])
        num_input_tokens += tokens_counter(input_text)
        num_output_tokens += tokens_counter(output_text)
    print(f'num input tokens of task {task_name}: {num_input_tokens}')
    print(f'num output tokens of task {task_name}: {num_output_tokens}')
    return num_input_tokens, num_output_tokens


if __name__ == '__main__':
    total_num_input_tokens = 0
    total_num_output_tokens = 0
    for task in ['ContinueWriting', 'HalluModified', 'QuestAnswer1Doc', 'QuestAnswer2Docs', 'QuestAnswer3Docs', 'Summary']:
        task_num_input_tokens, task_num_output_tokens = run(task)
        total_num_input_tokens += task_num_input_tokens
        total_num_output_tokens += task_num_output_tokens
    print(f'Total num input tokens: {total_num_input_tokens}')
    print(f'Total num output tokens: {total_num_output_tokens}')

    total_cost_input = total_num_input_tokens/1000000*0.15
    total_cost_output = total_num_output_tokens/1000000*0.6
    print(f'Total cost for input tokens: {total_cost_input}')
    print(f'Total cost for output tokens: {total_cost_output}')