import copy
import json
import os
from abc import ABC
from loguru import logger
from tqdm import tqdm
from threading import Lock
from src.llms.base import BaseLLM
from src.tasks.base import BaseTask
from src.retrievers.base import BaseRetriever
import concurrent.futures

class BaseEvaluator(ABC):
    def __init__(self,
                 task: BaseTask,
                 model: BaseLLM,
                 retriever: BaseRetriever,
                 dataset: list[dict],
                 save_context: bool = True,
                 data_path: str = None,
                 output_dir: str = './output',
                 save_infer_output: bool = True,
                 use_gt_ctx: bool = False,
                 inject_negative_ctx: bool = False,
                 num_threads: int = 40):
        """
        Args:
            model (BaseLLM): The large language model to be evaluated.
            retriever (BaseRetriever): The retriever to be evaluated.
            task (BaseTask): The task for evaluation.
            dataset (list[dict]): The dataset for evaluation.
            output_dir (str): The directory for result output and caching.
        """
        self.model = model
        self.retriever = retriever
        self.dataset = dataset
        self.task = task
        self.lock = Lock()
        self.num_threads = num_threads

        self.save_context = save_context
        self.data_path = data_path
        self.use_gt_ctx = use_gt_ctx
        self.inject_negative_ctx = inject_negative_ctx

        collection_name = self.retriever.collection_name
        similarity_top_k = self.retriever.similarity_top_k
        output_dir = os.path.join(output_dir, f'{collection_name}_top{similarity_top_k}_use_gt_ctx{int(use_gt_ctx)}_inject_negative_ctx{int(inject_negative_ctx)}_{model.__class__.__name__}')
        
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_path = os.path.join(
            output_dir, f'{self.task.__class__.__name__}_{model.params["model_name"]}_quest_eval_{self.task.quest_eval.params["model_name"]}.json'
        )
        self.save_infer_output = save_infer_output
        if self.save_infer_output:
            self.temp_infer_dir = os.path.join(output_dir, "temp", f'{self.task.__class__.__name__}_{model.params["model_name"]}')
            os.makedirs(self.temp_infer_dir, exist_ok=True)

        self.task.set_model(self.model, self.retriever)

    def task_generation(self, data_point):
        try:
            self.lock.acquire()
            retrieve_context = self.task.retrieve_docs(data_point)
            self.lock.release()
            data_point["retrieve_context"] = retrieve_context

        except Exception as e:
            logger.warning(repr(e))
            self.lock.release()
            data_point["retrieve_context"] = ''

        return self.task.model_generation(data_point)
    
    def retrieve4all(self, data_points, show_progress_bar=False, task_name=''):
        for data_point in (tqdm(data_points, desc=f"retrieving for task {task_name}") if show_progress_bar else data_points):
            try:
                retrieve_context = self.task.retrieve_docs(data_point, use_gt_ctx=self.use_gt_ctx, inject_negative_ctx=self.inject_negative_ctx)
                data_point["retrieve_context"] = retrieve_context
            except Exception as e:
                logger.warning(repr(e))
                data_point["retrieve_context"] = ''
        if self.save_context:
            output_file = self.data_path.replace('split_merged', f'split_{task_name}_w_ctx')
            # if not os.path.exists(output_file):
            with open(output_file, 'w', encoding='utf-8') as fw:
                json.dump(data_points, fw, ensure_ascii=False, indent=4)
        return data_points

    def batch_task_generation(self, data_points, batch_size=8, show_progress_bar=False, task_name=''):
        self.retrieve4all(data_points, show_progress_bar=show_progress_bar, task_name=task_name)
        batch_indexes = range(0, len(data_points), batch_size)
        for spilt_ids in (tqdm(batch_indexes, desc=f"inferencing for task {task_name}") if show_progress_bar else batch_indexes):
            batch_data_points = data_points[spilt_ids: spilt_ids+batch_size]
            self.task.batch_model_generation(batch_data_points)
            if self.save_infer_output:
                # save the infer output to resume for an abnormal termination
                for data_point in batch_data_points:
                    output_file = os.path.join(self.temp_infer_dir, f"{data_point['ID']}.json")
                    with open(output_file, 'w', encoding='utf-8') as fw:
                        json.dump(data_point, fw, ensure_ascii=False)
        return data_points

    def multithread_batch_scoring(self, dataset: list[dict], sort=True, show_progress_bar=False, contain_original_data=False) -> list[dict]:
        """Perform batch scoring on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """

        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        def process_data_point(data_point):
            if data_point['ID'] in saved_ids:
                return None  # Skip results that have already been evaluated and are valid
            try:
                generated_text = self.task_generation(data_point)
                # TODO fix bugs
                if generated_text == '","msg":"request openai failed"':
                    return None
                
                data_point["generated_text"] = generated_text
                result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                
                if contain_original_data:
                    result['original_data'] = data_point

                return result
            
            except Exception as e:
                logger.warning(repr(e))
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_results = list(tqdm(executor.map(process_data_point, dataset), total=len(dataset)))
        
        results.extend([result for result in future_results if result is not None])
        
        return sorted(results, key=lambda x: x['id']) if sort else results
    
    def batch_scoring(self, dataset: list[dict], sort=True, show_progress_bar=False, contain_original_data=False, batch_size=8, do_eval=True, re_quest_eval=False) -> list[dict]:
        """Perform batch scoring on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """
        id2result = dict()
        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
            id2result = {result['id']: result for result in results}
        else:
            results = []
            saved_ids = []
        remains_dataset = []
        for data_point in dataset:
            if data_point['ID'] not in saved_ids:
                remains_dataset.append(data_point)

        if do_eval and re_quest_eval:
            print('rerun RAGQuestEval.')
            for data_point in dataset:
                data_point['generated_text'] = id2result[data_point['ID']]['log']['generated_text'] if data_point['ID'] in id2result else ''
            dataset = self.task.batch_quest_eval(dataset, batch_size, show_progress_bar=show_progress_bar, task_name=self.task.__class__.__name__)
            new_results = []
            for data_point in (tqdm(dataset, desc=f"scoring for task {self.task.__class__.__name__}") if show_progress_bar else dataset):
                try:
                    result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                    id2result[data_point['ID']]['metrics']['QA_avg_F1'] = result['metrics']['QA_avg_F1']
                    id2result[data_point['ID']]['metrics']['QA_recall'] = result['metrics']['QA_recall']
                    id2result[data_point['ID']]['log']['evaluateDatetime'] = result['log']['evaluateDatetime']
                    id2result[data_point['ID']]['log']['quest_eval_save']['answers_gm4gt'] = result['log']['quest_eval_save']['answers_gm4gt']
                    new_results.append(id2result[data_point['ID']])
                except Exception as e:
                    logger.warning(repr(e))
                results = new_results
        elif remains_dataset:
            self.batch_task_generation(remains_dataset, batch_size, show_progress_bar=show_progress_bar, task_name=self.task.__class__.__name__)
            if do_eval and self.task.use_quest_eval and batch_size:
                remains_dataset = self.task.batch_quest_eval(remains_dataset, batch_size, show_progress_bar=show_progress_bar, task_name=self.task.__class__.__name__)
            if do_eval:
                for data_point in (tqdm(remains_dataset, desc=f"scoring for task {self.task.__class__.__name__}") if show_progress_bar else remains_dataset):
                    try:
                        result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                        if contain_original_data:
                            result['original_data'] = data_point
                        results.append(result)
                    except Exception as e:
                        logger.warning(repr(e))
        return sorted(results, key=lambda x: x['id']) if sort else results

    def save_output(self, output: dict, re_quest_eval=False) -> None:
        """Save evaluation results."""
        if re_quest_eval:
            self.output_path = self.output_path.replace('quest_eval', 're_quest_eval')
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort=True, show_progress_bar=False, contain_original_data=True, batch_size=8, do_eval=True, re_quest_eval=False) -> dict:
        """Run a complete evaluation.

        Args:            
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.
            contain_original_data (bool): Whether to include original data in the results for debugging.

        Returns:
            dict: Output dictionary contains fields such as: info, overall, results, etc.
        """
        info = {
            'task': self.task.__class__.__name__, 
            'llm': str(self.model.params),
            'quest_eval_llm': self.task.quest_eval.params['model_name']
        }
        results = self.batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data, batch_size=batch_size, do_eval=do_eval, re_quest_eval=re_quest_eval)
        if do_eval:
            valid_results = self.remove_invalid(results)
            try:
                overall = self.task.compute_overall(valid_results) if len(valid_results) > 0 else {}\
                # 保存用于评估的RAGQuestEval QA问答对
                if self.task.use_quest_eval:
                    self.lock.acquire()
                    self.task.quest_eval.save_quest_gt(self.task.__class__.__name__)
                    self.lock.release()
            
            except Exception as e:
                logger.warning(repr(e))
                overall = dict()

            self.save_output(output:={'info': info, 'overall': overall, 'results': results}, re_quest_eval=re_quest_eval)
            print(f'Output saved at {self.output_path}!')
            return output
        return

    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        """Remove invalid results from the list and return the cleaned results."""
        return [result for result in results if result['valid']]
