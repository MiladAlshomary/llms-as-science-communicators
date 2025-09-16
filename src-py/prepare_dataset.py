import os
import sys
os.environ['TRANSFORMERS_CACHE'] = '/mnt/swordfish-pool2/milad/hf-cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/swordfish-pool2/milad/hf-cache'
os.environ["WANDB_DIR"] = '/mnt/swordfish-pool2/milad/wandb-dir'
sys.path.append('./src-py')


from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import List, Dict
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import nltk
import re
import argparse


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import Dataset
import datasets

from trl import apply_chat_template

MAX_PAPER_TOKENS = 1000
MAX_CONV_TOKENS = 1000
# === STEP 1: Extract turns from full conversations ===

def extract_turns(tokenizer, example: Dict, role: str, max_num_turns=40) -> List[Dict]:
    paper = example["sc-intro"]
    turns = example["parsed_conv"]  # list of {"role": "journalist"/"researcher", "text": ...}

    pattern = r'\bDr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'


    examples = []
    context_so_far = []
    for i in range(min([len(turns), max_num_turns]) - 1):
        if turns[i + 1]["author"] != role:
            continue
        
        # Build the conversation context up to i
        context_lines = []
        so_far_number_of_tokens = 0
        for j in range(i, -1, -1):
            turn_content = re.sub(pattern, '[name]', turns[j]['text'])
            turn_tokens = tokenizer.tokenize(turn_content)
            if so_far_number_of_tokens + len(turn_tokens) > MAX_CONV_TOKENS:
                #print('Reached max numer of conversation tokens', '# turns', len(context_lines))
                #print(list(reversed(context_lines)))
                continue
                
            so_far_number_of_tokens+= len(turn_tokens)
                
            context_lines.append({'role': 'assistant' if turns[j]["author"] == role else 'user', 
                                  'content': '{}'.format(turns[j]["author"], turn_content)
            })

        input_prompt = {
            "paper_id": example['id'],
            "paper_title": example['pr-title'],
            "paper_text": paper,
            "prompt": list(reversed(context_lines)),
            "completion": [{'role': 'assistant', 
                            'content': '{}: {}'.format(turns[i+1]["author"],re.sub(pattern, '[name]', turns[i+1]['text']))}]
        }
        examples.append(input_prompt)
    return examples

# === STEP 2: Truncate paper/context to fit in max tokens ===

def truncate_to_max_tokens(paper_text, tokenizer, max_paper_tokens):
    # Sentence tokenization
    paper_sents = sent_tokenize(paper_text)


    # Truncate paper from the beginning (keep first sentences)
    truncated_paper = []
    paper_token_count = 0
    for sent in paper_sents:
        sent_tokens = tokenizer.tokenize(sent)
        if paper_token_count + len(sent_tokens) > max_paper_tokens:
            break
        truncated_paper.append(sent)
        paper_token_count += len(sent_tokens)

    # Join back into strings
    return " ".join(truncated_paper).strip()


# === STEP 3: Build formatted prompts for chat model ===

def build_chat_format(tokenizer, example, role="Journalist"):
    role_prompt = "You are a helpful and knowledgeable journalist asking questions about a scientific paper." \
        if role == "Journalist" else \
        "You are a helpful and expert researcher answering questions about your scientific paper."

    paper = truncate_to_max_tokens(
        example["paper_text"],
        tokenizer,
        max_paper_tokens=MAX_PAPER_TOKENS
    )

    paper_title = example['paper_title']
    
    user_prompt = f"[PAPER-TITLE]\n{paper_title}\n[PAPER]\n{paper}"


    #new_prompt = [[{'role': 'system', 'content': role_prompt}] + [{'role': 'user', 'content': user_prompt}]] + example['prompt']

    example['prompt'].insert(0, {'role': 'user', 'content': user_prompt})
    example['prompt'].insert(0, {'role': 'system', 'content': role_prompt})

    #example = apply_chat_template(example, tokenizer)
    
    return example #{"chat_text_tokenized": tokenized_prompt, "chat_text": prompt}

# === STEP 4: Wrap everything into a function for HF training ===

def prepare_dataset(tokenizer, full_data: List[Dict], role: str, max_num_turns=40) -> Dataset:
    
    # Step 1: extract role-specific training examples
    all_examples = []
    for example in tqdm(full_data):
        all_examples.extend(extract_turns(tokenizer, example, role=role, max_num_turns=max_num_turns))

    # Step 2: build HF Dataset
    dataset = Dataset.from_list(all_examples)

    # Step 3: format + tokenize using chat template
    tokenized = dataset.map(lambda ex: build_chat_format(tokenizer, ex, role=role), batched=False)

    return tokenized

def split_dataset_by_clm(all_data, clm, seed=123):
    import random
    from collections import defaultdict
    from datasets import Dataset, DatasetDict
    
    # 1. Group data by clm
    paper_to_examples = defaultdict(list)
    for example in tqdm(all_data):
        paper_to_examples[example[clm]].append(example)
    
    # 2. Get list of unique strings
    paper_ids = list(paper_to_examples.keys())
    random.seed(seed)  # reproducibility
    random.shuffle(paper_ids)
    
    # 3. Split paper_ids into train and eval
    split_ratio = 0.9
    split_idx   = int(len(paper_ids) * split_ratio)
    train_papers = set(paper_ids[:split_idx])
    eval_papers  = set(paper_ids[split_idx:])
    
    # 4. Reassemble datasets
    train_data = [ex for pid in train_papers for ex in paper_to_examples[pid]]
    eval_data = [ex for pid in eval_papers for ex in paper_to_examples[pid]]
    
    # 5. Wrap in HuggingFace datasets
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(eval_data)
    })

    return dataset

def process_dataset_for_training(tokenizer, dataset_path, output_path, max_num_turns=40):
    conv_dataset = datasets.load_from_disk(dataset_path)
    conv_dataset = datasets.Dataset.train_test_split(conv_dataset, test_size=0.1, seed=123)

    train_journalist_dataset = prepare_dataset(tokenizer, conv_dataset['train'], role="Journalist", max_num_turns=max_num_turns)
    train_researcher_dataset = prepare_dataset(tokenizer, conv_dataset['train'], role="Researcher", max_num_turns=max_num_turns)
    
    test_journalist_dataset = prepare_dataset(tokenizer, conv_dataset['test'], role="Journalist", max_num_turns=max_num_turns)
    test_researcher_dataset = prepare_dataset(tokenizer, conv_dataset['test'], role="Researcher", max_num_turns=max_num_turns)

    train_journalist_dataset = split_dataset_by_clm(train_journalist_dataset, 'paper_id')
    train_researcher_dataset = split_dataset_by_clm(train_researcher_dataset, 'paper_id')

    train_journalist_dataset.save_to_disk(output_path + '/train_journalist_ds')
    train_researcher_dataset.save_to_disk(output_path + '/train_researcher_ds')
    
    test_journalist_dataset.save_to_disk(output_path + '/test_journalist_ds')
    test_researcher_dataset.save_to_disk(output_path + '/test_researcher_ds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='prepare conversation data',
                    description='')

    parser.add_argument('ds_path')
    parser.add_argument('output_path')
    parser.add_argument('tokenizer_path')
    parser.add_argument('--max_num_turns', type=int, default=40)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token  # for older tokenizers
    process_dataset_for_training(tokenizer, args.ds_path, args.output_path, max_num_turns=args.max_num_turns)