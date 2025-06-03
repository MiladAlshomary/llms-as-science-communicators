import os
import sys
import re

os.environ['TRANSFORMERS_CACHE'] = '/mnt/swordfish-pool2/milad/hf-cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/swordfish-pool2/milad/hf-cache'
os.environ["WANDB_DIR"] = '/mnt/swordfish-pool2/milad/wandb-dir'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']='1'
sys.path.append('./src-py')

import datasets
import json
import os
import pandas as pd
import torch
import re
import random
import wandb
import tiktoken
from transformers import AutoTokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TextStreamer
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback

from datasets import Dataset

from accelerate import PartialState
device_string = PartialState().process_index

import argparse
from nltk import tokenize
import utils

import json
keys = json.load(open('../keys.json'))
huggingface_token = os.environ[keys['hf_token']]

def train_model(model, tokenizer, train_ds, valid_ds, output_path, run_name, eval_steps=200, max_length=2500, num_train_epochs=3, resume_from_checkpoint=False, extra_args=None):

    print(extra_args)
    print('=====================')
    # Shuffle the training set
    train_ds = train_ds.shuffle().select(range(40000))
    valid_ds = valid_ds.select(range(5000))
    #save the dataset we trained on

    if args.preprocess_data:
        train_ds = train_ds.map(lambda row: {'norm_prompt': normalize_chat_roles(row['prompt'])})
        valid_ds = valid_ds.map(lambda row: {'norm_prompt': normalize_chat_roles(row['prompt'])})
    
    wandb.init(project="training-llama-on-conversations", name=run_name)

    if 'qwen' in run_name:
        peft_config = LoraConfig(
                lora_alpha=extra_args.lora_alpha,
                lora_dropout=extra_args.lora_dropout,
                r=extra_args.lora_rank,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
        )
    else:
        peft_config = LoraConfig(
                lora_alpha=extra_args.lora_alpha,
                lora_dropout=extra_args.lora_dropout,
                r=extra_args.lora_rank,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
        )
    
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    print(model.print_trainable_parameters())

    sft_config = SFTConfig(
        run_name = run_name,
        output_dir= output_path + "/" + run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=extra_args.batch_size,
        resume_from_checkpoint=resume_from_checkpoint,
        gradient_accumulation_steps=extra_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        neftune_noise_alpha=5,
        logging_steps=5,
        eval_strategy="steps",
        learning_rate=extra_args.lr,
        bf16=True,
        tf32=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        max_grad_norm=0.3,
        warmup_ratio = 0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        push_to_hub=True,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        report_to="wandb",
        #dataset_text_field="chat_text",
        max_length=max_length,
        completion_only_loss=True,
        do_eval=True,
    )
    
    trainer = SFTTrainer(
        model,
        #tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        formatting_func=lambda example: format_example(example, tokenizer) if extra_args.preprocess_data else None,
        args=sft_config
    )

    # train_dataloader = trainer.get_train_dataloader()

    # batch = next(iter(train_dataloader))
    # print(batch)
    # print(batch['input_ids'].shape)
    # print(batch['input_ids'][0])
    # print(batch['labels'][0])
    # print(sum([1 if x != -100 else 0 for x in batch['labels'][0]]))


    #check validity of validation batches
    valid_dataloader = trainer.get_eval_dataloader()
    for batch in valid_dataloader:
        labels = batch["labels"]
        num_targets = (labels != -100).sum().item()
        if num_targets == 0:
            print(f"Supervised tokens in eval batch: {num_targets}")
            print("Exiting without training >>>>>>>>>>>>>>>>>>>>>")
            return
    
    
    trainer.train()
    trainer.save_model()
    
    #model.push_to_hub("miladalsh/conv-ft-llama3")
    #tokenizer.push_to_hub("miladalsh/conv-ft-llama3")

def format_example(example, tokenizer):
    prompt_messages = example["prompt"]
    completion_messages = example["completion"]

    # Optional: assertions to debug input structure
    assert prompt_messages[0]["role"] == "system"
    assert prompt_messages[1]["role"] == "user"
    assert completion_messages[-1]["role"] == "assistant"
    assert all(
        prompt_messages[i]["role"] != prompt_messages[i+1]["role"]
        for i in range(len(prompt_messages)-1)
    ), f"Non-alternating roles in: {[m['role'] for m in prompt_messages]}"

    # Combine prompt and completion
    full_messages = prompt_messages + completion_messages

    # Apply chat template
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)

    # Tokenize
    full_tokens = tokenizer(full_text, return_tensors="pt", padding=False, truncation=True)
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True)

    input_ids = full_tokens["input_ids"][0]
    labels = input_ids.clone()

    # Mask prompt tokens
    labels[:prompt_tokens["input_ids"].shape[1]] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def normalize_chat_roles(messages):
    if not messages:
        return messages

    normalized = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == normalized[-1]["role"]:
            normalized[-1]["content"] += "\n" + msg["content"]
        else:
            normalized.append(msg)
    return normalized

def train_main(conv_ds_path, output_path, model_name, run_name, num_epochs, resume_from_checkpoint, extra_args):
    conv_ds = datasets.load_from_disk(conv_ds_path)
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

        
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 token=huggingface_token,
                                                 device_map="auto",
                                                 attn_implementation="flash_attention_2" if extra_args.use_flash_attention_2 else None,
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config)

    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_model(model, tokenizer, conv_ds['train'], conv_ds['test'], output_path, run_name=run_name, eval_steps=extra_args.eval_steps, num_train_epochs=num_epochs, resume_from_checkpoint=resume_from_checkpoint, extra_args=extra_args)


#======== Commands ====
# python training_llm_on_conversations.py prepare_data /mnt/swordfish-pool2/milad/communicating-science-to-the-public/llama3-final-conv-ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/llama3-3-final-conv-ds-preprocessed
# 
# python training_llm_on_conversations.py prepare_data /mnt/swordfish-pool2/milad/communicating-science-to-the-public/gpt3-final-conv-ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/gpt3-final-conv-ds-preprocessed
#
# CUDA_VISIBLE_DEVICES=2 python training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/llama3-3-final-conv-ds-preprocessed /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 3 --run_name llama3-trained-on-llama3-for-3-epochs /mnt/swordfish-pool2/milad/conda-envs/huggingface-tlr/lib/p
#
# python training_llm_on_conversations train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/gpt3-final-conv-ds-preprocessed /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 3 --run_name llama3-trained-on-gpt3-for-3-epochs
#

# python training_llm_on_conversations prepare_data /mnt/swordfish-pool2/milad/communicating-science-to-the-public/llama3-final-conv-ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/llama3-3-final-conv-ds-for-journalist-preprocessed --role_name journalist

# python training_llm_on_conversations.py prepare_data /mnt/swordfish-pool2/milad/communicating-science-to-the-public/gpt3-final-conv-ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/gpt3-final-conv-ds-for-journalist-preprocessed --role_name journalist

# CUDA_VISIBLE_DEVICES=2 python training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/llama3-3-final-conv-ds-for-journalist-preprocessed /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 1 --run_name llama3-trained-journalist-on-llama3-for-1-epochs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='training conversational model',
                    description='')

    parser.add_argument('action')           # positional argument
    parser.add_argument('conv_ds_path')      # option that takes a value
    parser.add_argument('output_path') 
    parser.add_argument('--training_epochs', type=int, default=3) 
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--preprocess_data', action='store_true', default=False)
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--role_name', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--use_flash_attention_2', action='store_true', default=False)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    #LoRA parameters
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    

    args = parser.parse_args()

    
    if args.action == 'prepare_data':
        prepare_dataset_for_training(args.conv_ds_path, args.output_path, args.role_name)

    elif args.action == 'train':           
        train_main(args.conv_ds_path, args.output_path, args.model_name, args.run_name, args.training_epochs, args.resume_from_checkpoint, args)