import prompts
import random
import os

import datasets
from tabulate import tabulate
import numpy as np
import pandas as pd

from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
from openai import OpenAIError  # Base class for OpenAI exceptions
import openai
import json
from tqdm import tqdm
from openai.lib._pydantic import to_strict_json_schema

keys = json.load(open('./keys.json'))

eval_client = OpenAI(base_url="http://localhost:9988/v1", api_key="-")
openai_eval_client = OpenAI(api_key=keys['OPENAI_API_KEY'])
anthro_eval_client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=keys['ANTHRO-API-KEY'])


class ThreePoints(int, Enum):
    one   = 1
    two   = 2
    three = 3

class FivePoints(int, Enum):
    one   = 1
    two   = 2
    three = 3
    four  = 4
    five  = 5

class ThreePointsScoring(BaseModel):
    reasons: str
    score: ThreePoints

class FivePointsScoring(BaseModel):
    reasons: str
    score: FivePoints


def evaluate_communicative_quality(dataset, eval_prompt, evaluator_name):

    json_schema = to_strict_json_schema(FivePointsScoring) if eval_prompt['scoring_scheme'] == '5_points' else to_strict_json_schema(ThreePointsScoring)
    eval_scores = []
    for row in tqdm(dataset):
        instruction = '{}\n\n{}'.format(eval_prompt['instruction'], '\n\n'.join(["{}: {}".format(input_name, row[input_val]) for input_name, input_val in eval_prompt['inputs'].items()]))
        if 'gpt' in evaluator_name:        
            completion = openai_eval_client.chat.completions.create(
                model=evaluator_name,
                messages=[
                    {
                        "role": "user",
                        "content": instruction,
                    },
                ],
                response_format={"type": "json_schema", "json_schema": {"name": eval_prompt['strategy_name'], "schema": json_schema}},
            )
            score = completion.choices[0].message.content
        else:
            completion = eval_client.chat.completions.create(
                model=evaluator_name,
                messages=[
                    {
                        "role": "user",
                        "content": instruction,
                    }
                ],
                extra_body={"guided_json": json_schema}
            )
            score = completion.choices[0].message.reasoning_content
            
        try:
            eval_scores.append(json.loads(score))           
        except openai.BadRequestError as e:
            print(f"BadRequestError: {e}")  # Log the error message
            eval_scores.append({'reasons':'no reason', 'score':1})
        except OpenAIError as e:
            print(f"OpenAI API Error: {e}")  # Catch all OpenAI-related errors
            eval_scores.append({'reasons':'no reason', 'score':1})
        except Exception as e:
            print(f"Unexpected error: {e}")  # Catch any other unexpected errors
            eval_scores.append({'reasons':'no reason', 'score':1})

    dataset = dataset.add_column('{}_scoring_parsed'.format(eval_prompt['strategy_name']), eval_scores)
    return dataset
    
def get_llm_avg_scores(llm_eval, prompts_to_eval):
    scoring_matrix = np.array([[item['{}_scoring_parsed'.format(prompt['strategy_name'])]['score'] for prompt in prompts_to_eval if '{}_scoring_parsed'.format(prompt['strategy_name']) in item]
                     for item in llm_eval]).astype(int)
    
    scoring_matrix[scoring_matrix == None] = 0.00
    avg_scoring = np.mean(scoring_matrix, axis=0).astype(float)
    avg_scoring = np.around(avg_scoring, 2)
    return list(avg_scoring) + [round(np.mean(avg_scoring), 2)]


def llm_based_evaluation(prompts_to_eval, datasets_to_eval, force_generation=False, evaluator_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
    
    llm_eval_results = {}
    for name, ds_nd_path in datasets_to_eval.items():
        ds = ds_nd_path[0]
        ds_path = ds_nd_path[1]
        if os.path.exists(ds_path + '/ds_eval/' + evaluator_name.split('/')[0]) and not force_generation:
            print('Loading {} from already saved file'.format(ds_path + '/ds_eval/' + evaluator_name.split('/')[0]))
            llm_eval_results[name] = datasets.load_from_disk(ds_path + '/ds_eval/' + evaluator_name.split('/')[0])

        else:
            for eval_prompt in prompts_to_eval:
                ds = evaluate_communicative_quality(ds, eval_prompt, evaluator_name=evaluator_name)
            
            llm_eval_results[name] = ds
            ds.save_to_disk(ds_path + '/ds_eval/' + evaluator_name.split('/')[0])

    print(tabulate(
        [[name] + get_llm_avg_scores(res, prompts_to_eval) for name, res in llm_eval_results.items()],
        headers=['#'] + [p['strategy_name'] for p in prompts_to_eval] + ['Avg']
    ))

    return llm_eval_results