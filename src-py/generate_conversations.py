import os
import sys
import json

os.environ['TRANSFORMERS_CACHE'] = '/mnt/swordfish-pool2/milad/hf-cache-new'
os.environ['HF_DATASETS_CACHE'] = '/mnt/swordfish-pool2/milad/hf-cache-new'
sys.path.append('../src-py')

keys = json.load(open('/local/nlp/milad/code/llms-as-science-communicators/keys.json'))
for key, val in keys.items():
    os.environ[key] = val

import json
import numpy as np
import pandas as pd
import copy
import re
from collections import Counter

import utils
import prompts
import random
#import datadreamer_generation

from tabulate import tabulate
import tiktoken
import argparse
import datasets
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login

login(os.environ['hf_token'])

output_dir = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/'
models_folder = "/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/"
gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def generate_conversations_from_fixed_questions(ds, output_path, researcher_base_model_name, researcher_adapter_name=""):
    if researcher_adapter_name == "": # It's baseline generation
        researcher_prompt = """
            You are a helpful and expert researcher answering questions about your scientific paper. 
            1. You are excellent at communicating your research in a simple and everyday life language
            2. You know how to communicate the social impact of your research.
            3. You know how to put your research in the proper scientific context
            
            Limit your answer to one single paragraph
        """
    else:
        researcher_prompt = "You are a helpful and expert researcher answering questions about your scientific paper."

    print(researcher_prompt)
    
    researcher_model, tokenizer = utils.load_model_with_adapter(researcher_base_model_name, researcher_adapter_name, device_map="cuda:0")
    researcher_pipeline = pipeline("text-generation", model=researcher_model, tokenizer=tokenizer, batch_size=12)

    resulted_ds = utils.construct_full_dialogue_from_fixed_questions(ds, researcher_pipeline, paper_title_clm='paper_title', paper_text_clm='paper_text', output_clm = 'fixed_journalist_conv', researcher_prompt = researcher_prompt)
    
    resulted_ds.save_to_disk(output_path)
    
def generate_conversations_interactively(ds, output_path, journalist_base_model_name, researcher_base_model_name, journalist_adapter_name="", researcher_adapter_name="", baseline_researcher=False, max_rounds=5):

    if researcher_adapter_name == "": # It's baseline generation
        if baseline_researcher:
            researcher_prompt = """You are a researcher answering questions about your scientific paper"""
        else:
            researcher_prompt = """
                You are a helpful and expert researcher answering questions about your scientific paper. 
                1. You are excellent at communicating your research in a simple and everyday life language
                2. You know how to communicate the social impact of your research.
                3. You know how to put your research in the proper scientific context
                
                Limit your answer to one single paragraph
            """
    else:
        researcher_prompt = "You are a helpful and expert researcher answering questions about your scientific paper."


    
    if journalist_adapter_name == "":
        journalist_prompt = """
            You are a helpful and knowledgeable journalist asking questions about a scientific paper. 
            1. Your questions encourage the researcher to place their paper in a proper societal and scientific context to the greatest possible degree.
            2. Your questions focus on topics in the paper that are novel and have unexpected results.
            3. Your questions follow up on the researcher's answers, trying to clarify unexplained technical terms in everyday language.
            
            Ask a single new question or a follow-up question on the conversation.
        """
        #journalist_prompt="You are a helpful and knowledgeable journalist asking questions about a scientific paper."
        
    else:
        journalist_prompt="You are a helpful and knowledgeable journalist asking questions about a scientific paper."

    print(researcher_prompt)
    print(journalist_prompt)

    print('Journalist is {} base with adapter {}'.format(journalist_base_model_name, journalist_adapter_name))
    print('Researcher is {} base with adapter {}'.format(researcher_base_model_name, researcher_adapter_name))
    journalist_model, journalist_tokenizer = utils.load_model_with_adapter(journalist_base_model_name, journalist_adapter_name, device_map="cuda:0")
    researcher_model, researcher_tokenizer = utils.load_model_with_adapter(researcher_base_model_name, researcher_adapter_name, device_map="cuda:1")
    journalist_pipeline = pipeline("text-generation", model=journalist_model, tokenizer=journalist_tokenizer, batch_size=12)
    researcher_pipeline = pipeline("text-generation", model=researcher_model, tokenizer=researcher_tokenizer, batch_size=12)

    resulted_ds = utils.construct_full_dialogue(ds, journalist_pipeline, researcher_pipeline, max_journalist_turn_tokens=200, max_researcher_turn_tokens=500, journalist_prompt=journalist_prompt, researcher_prompt=researcher_prompt, max_rounds=max_rounds)
    
    resulted_ds.save_to_disk(output_path)
    
def generate_full_conversations(ds, output_path, base_model_name):
    all_prompts = utils.get_prompt_compositions()
    used_prompt = all_prompts[4]
    used_prompt['inputs']['Scientific paper'] = 'paper_text'

    tokenizer =  llama_tokenizer if 'llama3' in base_model_name else tiktoken
    resulted_ds = datadreamer_generation.generate_conversation(output_path, base_model_name, ds, used_prompt, tokenizer, max_input_tokens=1200, max_new_tokens=1000)
    resulted_ds.save_to_disk(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='generate conversations',
                    description='')

    parser.add_argument('ds_path')
    parser.add_argument('output_path')
    parser.add_argument('--journalist_base_model_name')
    parser.add_argument('--researcher_base_model_name')
    parser.add_argument('--full_conv', action='store_true', default=False)
    parser.add_argument('--fixed_questions', action='store_true', default=False)
    parser.add_argument('--baseline_researcher', action='store_true', default=False) # If set to true then we are using a baseline researcher
    parser.add_argument('--journalist_adapter_name', type=str, default="") 
    parser.add_argument('--researcher_adapter_name', type=str, default="")
    parser.add_argument('--max_rounds', type=int, default=5)

    args = parser.parse_args()

    sample_dataset = datasets.load_from_disk(args.ds_path)

    if args.fixed_questions:
        generate_conversations_from_fixed_questions(sample_dataset, args.output_path, args.researcher_base_model_name, researcher_adapter_name=args.researcher_adapter_name)
        exit()
    
    if args.full_conv:
        generate_full_conversations(sample_dataset, args.output_path, args.base_model_name)
    else:
        generate_conversations_interactively(sample_dataset, args.output_path, args.journalist_base_model_name, args.researcher_base_model_name, journalist_adapter_name=args.journalist_adapter_name, researcher_adapter_name=args.researcher_adapter_name, baseline_researcher=args.baseline_researcher, max_rounds=args.max_rounds)
