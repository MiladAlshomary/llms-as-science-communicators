import datasets
import json
import os
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from datadreamer import DataDreamer
from datadreamer.llms import HFTransformers, ParallelLLM, OpenAI
from datadreamer.steps import DataFromPrompt, ProcessWithPrompt,  HFHubDataSource, DataSource, zipped, concat

from datasets import Dataset, Sequence, Value
from functools import partial

import prompts
import re
from utils import truncate_text

def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds

def generate_fewshot_pr_summaries(model, ds, ds_path, source_clm='sc-abstract', hub_name=None):
    pass

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