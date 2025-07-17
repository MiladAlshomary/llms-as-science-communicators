import os 
os.environ['LC_ALL']='C.UTF-8'

from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers, ParallelLLM, OpenAI
from datadreamer.steps import DataFromPrompt, ProcessWithPrompt,  HFHubDataSource, DataSource, zipped, concat
from datadreamer.trainers import TrainHFFineTune
from peft import LoraConfig

from datasets import *
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import QuantoConfig

from transformers import BitsAndBytesConfig

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import re
import json
import tiktoken
from utils import parse_conversation, truncate_text
import prompts
import os

q=BitsAndBytesConfig(load_in_8bit=True)

# llama3 = ParallelLLM(HFTransformers("meta-llama/Meta-Llama-3-8B-Instruct", device=0, quantization_config=q, dtype=torch.bfloat16), 
#                      HFTransformers("meta-llama/Meta-Llama-3-8B-Instruct", device=1, quantization_config=q, dtype=torch.bfloat16))

deepseek_r1 = OpenAI(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", api_key='', base_url="http://localhost:9988/v1")

llama3 = OpenAI(model_name="meta-llama/Meta-Llama-3-8B-Instruct", api_key='', base_url="http://localhost:9977/v1")



gpt3 = OpenAI(model_name="gpt-3.5-turbo", api_key=os.environ['OPENAI_API_KEY'])
gpt4 = OpenAI(model_name="gpt-4", api_key=os.environ['OPENAI_API_KEY'])


gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
deepseek_r1_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")

available_models = {
    'llama3': llama3,
    'gpt-3': gpt3,
    'gpt-4' : gpt4,
    'deepseek-r1': deepseek_r1
}

available_tokenizers = {
    'llama3': llama_tokenizer,
    'gpt-3': gpt_tokenizer,
    'gpt-4' : gpt_tokenizer,
    'deepseek-r1': deepseek_r1_tokenizer
}

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def generate_conversation(data_path, model_name, dataset, prompt, encoding, max_input_tokens=15000, max_new_tokens=800, adapter_name='', tmp=None, top_p=None):

    if model_name in available_models:
        model = available_models[model_name]
    else:
        model = HFTransformers(model_name, device=0, quantization_config=q, dtype=torch.bfloat16)

    if adapter_name != '':
        model = HFTransformers("meta-llama/Meta-Llama-3-8B-Instruct", device=0, adapter_name=adapter_name, quantization_config=q, dtype=torch.bfloat16)
        #llm_2 = HFTransformers("meta-llama/Meta-Llama-3-8B-Instruct", device=1, adapter_name=adapter_name, quantization_config=q, dtype=torch.bfloat16)        
        #model = ParallelLLM(llm_1, llm_2)
    
    with DataDreamer(data_path + prompt['strategy_name']):
        datasource = DataSource('documents', dataset)
        datasource = datasource.map(lambda row: {'inputs': '\n\n'.join(["{}:\n{}".format(input_name, row[input_val]) for input_name, input_val in prompt['inputs'].items()])}, auto_progress=False)
        datasource = datasource.map(lambda row: {'inputs_truncated': truncate_text(encoding, row['inputs'], max_input_tokens)})
        #datasource = datasource.map(lambda row: {'inputs_len': len(encoding.encode(row['inputs']))})
        #datasource = datasource.filter(lambda row: row['inputs_len'] < max_input_tokens)

        ds_convs = ProcessWithPrompt(
          "generate conversations",
          inputs={"inputs": datasource.output["inputs_truncated"]},
          args={
             "llm": model,
             "n": 1,
             "temperature": tmp,
             "top_p": top_p,
             "max_new_tokens":max_new_tokens,
             "instruction": prompt['instruction']
          }, # Tem and top_p taken from here https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
          outputs={"generations": "conversation"},
        ).select_columns(["conversation"])
        
        zipped_step = zipped(datasource, ds_convs)
        zipped_step = zipped_step.map(lambda row: parse_conversation(row))

        results_iter = zipped_step.output.dataset
        results_ds   = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter))

        return results_ds

def parse_json_evaluation(row, clm):
    try:
        if not row[clm].startswith("{"):
            row[clm] = row[clm].split("\n")[-1]
            

        json_eval = json.loads(row[clm])
        return {
            clm + '_parsed': json_eval
        }
    except:
        print('Exception in parsing llm-based scoring >>>>> ', row[clm])
        return {
            clm + '_parsed': {'score': 5, 'reason': ''}
        }

def deepseek_parse_json_evaluation(row, clm):
    try:
        # For generation of DeepSeek we need to split over </think> tag and remove it
        eval_str = row[clm].split('</think>')[-1].strip()
        #eval_str = re.split(r'\n\n', eval_str)[-1].strip()
        #eval_str_found = re.findall(r"```json[^`]*```", eval_str)
        eval_str_found = re.findall(r'{\s*"reasons": "[^"]+",\s*"score": "?\d"?\s*}', eval_str)
        
        if len(eval_str_found) == 0:
            print("Json Eval Not Found >>>>> ", eval_str)
            return {
                clm + '_parsed': {'score': 1, 'reasons': ''}
            }
        
        eval_str = eval_str_found[0].replace('```json', '').replace('```', '')
        json_eval = json.loads(eval_str)

        output_json_eval = {}
        output_json_eval['reasons'] = str(json_eval['reasons']) if 'reasons' in json_eval else '' # ensure that reasons given are string
        output_json_eval['score']   = str(json_eval['score']) if 'score' in json_eval else 1 
        return {
            clm + '_parsed': output_json_eval
        }
    except:
        print('Exception in parsing llm-based scoring >>>>> ', eval_str)
        return {
            clm + '_parsed': {'score': 1, 'reasons': ''}
        }

def evaluate_conversation(data_path, model_name, dataset, encoding, max_input_tokens=15000):
    
    model  = available_models[model_name]
    prompt = prompts.llm_based_evaluation_prompt 
    with DataDreamer(data_path):
        datasource = DataSource('documents', dataset)
        datasource = datasource.map(lambda row: {'inputs': '\n\n'.join(["{}: {}".format(input_name, row[input_val]) for input_name, input_val in prompt['inputs'].items()])}, auto_progress=False)
        datasource = datasource.map(lambda row: {'inputs_len': len(encoding.encode(row['inputs']))})
        datasource = datasource.filter(lambda row: row['inputs_len'] < max_input_tokens)

        
        ds_convs = ProcessWithPrompt(
          "evaluate conversations",
          inputs={"inputs": datasource.output["inputs"]},
          args={
             "llm": model,
             "n": 1,
             "instruction": prompt['instruction']
          },
          outputs={"generations": "scoring"},
        ).select_columns(["scoring"])
        
        zipped_step = zipped(datasource, ds_convs)
        zipped_step = zipped_step.map(lambda row: parse_json_evaluation(row))

        results_iter = zipped_step.output.dataset
        results_ds   = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter))

        return results_ds

def evaluate_conversation_new(data_path, model_name, dataset, eval_prompts=[], max_input_tokens=15000):
    
    model  = available_models[model_name]
    encoding = available_tokenizers[model_name]

    for eval_prompt in eval_prompts:
        prompt_name = eval_prompt['strategy_name']
        print(prompt_name)
        print(eval_prompt['instruction'])
        with DataDreamer(data_path):
            
            datasource = DataSource('{}_documents'.format(prompt_name), dataset)
            datasource = datasource.map(lambda row: {prompt_name + '_inputs': '\n\n'.join(["{}: {}".format(input_name, row[input_val]) for input_name, input_val in eval_prompt['inputs'].items()])}, auto_progress=False)
            datasource = datasource.map(lambda row: {'inputs_len': len(encoding.encode(row[prompt_name + '_inputs']))})
            
            xxx     = Dataset.from_generator(partial(gen_from_iterable_dataset, datasource.output.dataset))
            print('Max input:', max(xxx['inputs_len']), '#', len(xxx['inputs_len']))
            datasource = datasource.filter(lambda row: row['inputs_len'] < max_input_tokens)
            xxx     = Dataset.from_generator(partial(gen_from_iterable_dataset, datasource.output.dataset))
            print('Max input: ', max(xxx['inputs_len']), '#', len(xxx['inputs_len']))
            
            ds_convs = ProcessWithPrompt(
              "{}_evaluate_conversations".format(prompt_name),
              inputs={"inputs": datasource.output[prompt_name + '_inputs']},
              args={
                 "llm": model,
                 "n": 1,
                 "top_p":0.01,
                 "temprature":0.0,
                 "instruction": eval_prompt['instruction']
              },
              outputs={"generations": "{}_scoring".format(prompt_name)},
            ).select_columns(["{}_scoring".format(prompt_name)])
            
            zipped_step = zipped(datasource, ds_convs)
            zipped_step = zipped_step.map(lambda row: deepseek_parse_json_evaluation(row, "{}_scoring".format(prompt_name)) if 'deepseek' in model_name else parse_json_evaluation(row, "{}_scoring".format(prompt_name)))
            results_iter= zipped_step.output.dataset
            dataset     = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter))

    return dataset

def evalaute_pr_article(data_path, model_name, dataset, encoding, max_input_tokens=15000):
    
    model  = available_models[model_name]
    prompt = prompts.llm_evaluate_pr_article 
    with DataDreamer(data_path):
        datasource = DataSource('documents', dataset)
        datasource = datasource.map(lambda row: {'inputs': '\n\n'.join(["{}: {}".format(input_name, row[input_val]) for input_name, input_val in prompt['inputs'].items()])}, auto_progress=False)
        datasource = datasource.map(lambda row: {'inputs_len': len(encoding.encode(row['inputs']))})
        datasource = datasource.filter(lambda row: row['inputs_len'] < max_input_tokens)

        
        ds_convs = ProcessWithPrompt(
          "generate conversations",
          inputs={"inputs": datasource.output["inputs"]},
          args={
             "llm": model,
             "n": 1,
             "instruction": prompt['instruction']
          },
          outputs={"generations": "scoring"},
        ).select_columns(["scoring"])
        
        zipped_step = zipped(datasource, ds_convs)
        zipped_step = zipped_step.map(lambda row: parse_json_evaluation(row))

        results_iter = zipped_step.output.dataset
        results_ds   = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter))

        return results_ds

def extract_topics_from_pr(data_path, dataset, encoding, max_input_tokens=15000):
    model = available_models['llama3']
    prompt = prompts.extracting_topic_prompt
    with DataDreamer(data_path):
        datasource = DataSource('documents', dataset)
        datasource = datasource.map(lambda row: {'inputs': '\n\n'.join(["{}: {}".format(input_name, row[input_val]) for input_name, input_val in prompt['inputs'].items()])}, auto_progress=False)
        datasource = datasource.map(lambda row: {'inputs_len': len(encoding.encode(row['inputs']))})
        datasource = datasource.filter(lambda row: row['inputs_len'] < max_input_tokens)
    
        
        ds_convs = ProcessWithPrompt(
          "extract_topics",
          inputs={"inputs": datasource.output["inputs"]},
          args={
             "llm": model,
             "n": 1,
             "instruction": prompt['instruction']
          },
          outputs={"generations": "topics"},
        ).select_columns(["topics"])
        
        zipped_step = zipped(datasource, ds_convs)
        zipped_step = zipped_step.map(lambda row: {"parsed-topics": '\n'.join([x.strip() for x in row['topics'].split('\n')[1:-1] if len(x.strip()) > 0])})
    
        results_iter = zipped_step.output.dataset
        results_ds   = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter))
    return results_ds

def summarize_pr_article(model, ds, ds_path, source_clm='pr-article', hub_name=None):
    instruction = """
    Please summarize the following article into a single paragraph.
    """

    with DataDreamer(ds_path):
        #make sure to cut the 
        datasource = DataSource('original_ds', ds)

        ds_pr_summaries = ProcessWithPrompt(
          "article-summary",
          inputs={"inputs": datasource.output[source_clm]},
          args={
             "llm": model,
             "n": 1,
             "instruction": instruction
          },
          outputs={"generations": "pr-summary"},
        ).select_columns(["pr-summary"])

        zipped_step = zipped(datasource, ds_pr_summaries)
        
        if hub_name != None:
            zipped_step.publish_to_hf_hub(hub_name)
            
        results_iter = zipped_step.output.dataset
        results_ds   = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter), features=results_iter.features)
        #pandas_df  = Dataset.from_generator(partial(gen_from_iterable_dataset, results_ds), features=results_ds.features).to_pandas()
        
        return results_ds

def parse_pr_article(row):
    gen_pr = row['gen-pr']
    #print(gen_pr)
    start_of_pr = re.search(r"Press Release Article", gen_pr, re.MULTILINE)
    if start_of_pr is None:
        pr_article = gen_pr
        print('Failed to find PR Article in the output')
    else:
        pr_article = gen_pr[start_of_pr.start(0) + len("Press Release Article"):].strip()

    return {"parsed-pr-article": pr_article}


def generate_pr_articles(model, ds, ds_path, encoding, prompt, max_input_tokens=5000, hub_name=None):
        

    with DataDreamer(ds_path + prompt['strategy_name']):
        #make sure to cut the 
        datasource = DataSource('original_ds', ds)

        datasource = datasource.map(lambda row: {'inputs': '\n\n'.join(["{}: {}".format(input_name, row[input_val]) for input_name, input_val in prompt['inputs'].items()])}, auto_progress=False)
        datasource = datasource.map(lambda row: {'inputs_truncated': truncate_text(encoding, row['inputs'], max_input_tokens)})
        #datasource = datasource.map(lambda row: {'inputs_len': len(encoding.encode(row['inputs']))})
        #datasource = datasource.filter(lambda row: row['inputs_len'] < max_input_tokens)

        print(prompt['instruction'])
        ds_desc = ProcessWithPrompt(
          "pr-writings",
          inputs={"inputs": datasource.output['inputs_truncated']},
          args={
             "llm": model,
             "n": 1,
             "instruction": prompt['instruction'],
          },
          outputs={"generations": "gen-pr"},
        ).select_columns(["gen-pr"])


        zipped_step = zipped(datasource, ds_desc)
        zipped_step = zipped_step.map(lambda row: parse_pr_article(row))
        
        if hub_name != None:
            zipped_step.publish_to_hf_hub(hub_name)
            
        results_iter = zipped_step.output.dataset
        results_ds   = Dataset.from_generator(partial(gen_from_iterable_dataset, results_iter), features=results_iter.features)
        #pandas_df  = Dataset.from_generator(partial(gen_from_iterable_dataset, results_ds), features=results_ds.features).to_pandas()
        
        return results_ds