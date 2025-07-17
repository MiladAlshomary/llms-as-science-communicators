import datasets
import json
import os
import pandas as pd
import torch
import re
import ast 
from prompts import *
from openai import OpenAI
from transformers import pipeline, TextStreamer
import torch
from transformers import BitsAndBytesConfig
from rouge_score import rouge_scorer
from evaluate import load
bertscore = load("bertscore")
import numpy as np
import nltk
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM

q=BitsAndBytesConfig(load_in_8bit=True)

#need to set openAI key here
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def count_tokens(tokenizer, text):
        """Helper function to count tokens in text."""
        return len(tokenizer.tokenize(text))

def truncate_text(tokenizer, text, max_tokens):
    """Truncate text to fit within max_tokens while keeping meaning."""
    
    tokens = tokenizer.tokenize(text) if hasattr(tokenizer, 'tokenize') else tokenizer.encode(text)
        
    if len(tokens) <= max_tokens:
        return text

    # Find last complete sentence boundary
    sentences = nltk.tokenize.sent_tokenize(text)
    # Rebuild text up to last full sentence
    result = ''
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence) if hasattr(tokenizer, 'tokenize') else tokenizer.encode(sentence)
        if current_tokens + len(sentence_tokens) > max_tokens:
            break
        result += sentence.strip() + ' '
        current_tokens += len(sentence_tokens)

    return result.strip()

def is_paragraph_complete(paragraph: str) -> bool:
    """
    Checks if the paragraph ends with a sentence-ending punctuation mark.
    Accepts ., !, or ? possibly followed by quotes or brackets.
    """
    # Normalize whitespace at the end
    paragraph = paragraph.strip()

    # Regex: sentence should end in . ! ? possibly followed by quotes or brackets
    pattern = r"[.?!…](['”\"\)]*)\s*$"
    return bool(re.search(pattern, paragraph))

def extract_complete_paragraphs(text: str, max_paragraphs: int = 3) -> str:
    """
    Extracts up to `max_paragraphs` that end properly from the given text.
    
    Parameters:
    - text (str): Model-generated output
    - max_paragraphs (int): Maximum number of paragraphs to include (optional)
    
    Returns:
    - str: Cleaned text of fully complete paragraphs only
    """
    # Split and clean
    paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    
    complete_paragraphs = []
    for paragraph in paragraphs:
        if is_paragraph_complete(paragraph):
            complete_paragraphs.append(paragraph)
            if len(complete_paragraphs) >= max_paragraphs:
                break
        else:
            break  # Stop at the first incomplete paragraph

    if len(complete_paragraphs) == 0: # we failed in cutting
        return text
    else:
        return '\n\n'.join(complete_paragraphs)


def load_model_with_adapter(base_model_path, adapter_path="", device_map="auto"):
    

    if adapter_path != "":
        # Load LoRA config
        peft_config = PeftConfig.from_pretrained(adapter_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        
        # Load model with LoRA adapter applied
        base_model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        return base_model, tokenizer
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token  # set pad token
        return base_model, tokenizer
        
    

def get_prompt_compositions():
    prompt = composite_prompt

    #initialize all possible combinations of instructions
    all_prompts = []
    for r_k, r_g in researcher_guidelines.items():
        for j_k, j_g in journalist_guidelines.items():
            tmp_prompt = json.loads(json.dumps(prompt)) # as alternative of deepcopy
            tmp_prompt['strategy_name'] = tmp_prompt['strategy_name'] + '-' + r_k + '-' + j_k
            tmp_prompt['instruction']   = tmp_prompt['instruction'].replace('[journalist-guidelines]', j_g)
            tmp_prompt['instruction']   = tmp_prompt['instruction'].replace('[researcher-guidelines]', r_g)
            if j_k == 'pr-guided':
                tmp_prompt['inputs']['Journalistic report'] = 'pr-article'
            if j_k == 'pr-topic-guided':
                tmp_prompt['inputs']['Topics'] = 'parsed-topics'

            all_prompts.append(tmp_prompt)
    return all_prompts
        
def build_model_context(row, encoding, context=['abstract', 'introduction'], max_token_number=1000):
    #for the SciNews corpus we don't have yet section-names, so we return first 1k tokens as context
    if row['sc-section_names'] is None or len(row['sc-section_names']) == 0:
        #print('Empty section names')
        sents = nltk.sent_tokenize(row['sc-article'])
        out_sents = ''
        i = 0
        so_far_num_tokens = 0
        while i < len(sents) and so_far_num_tokens < max_token_number:
            out_sents+= sents[i] + ' '
            i+=1
            so_far_num_tokens = len(encoding.encode(out_sents))
        return out_sents
        
    section_names = [x.lower() for x in row['sc-section_names']]
    context_to_idx = {}
    for c in context:
        for i, s in enumerate(section_names):
            if c in s:
                context_to_idx[c] = i
                break

    found_sections = {s[0]: row['sc-sections'][s[1]] for s in context_to_idx.items()}
    
    if len(found_sections) == 0: #if we fail in finding the needed sections, we return th firs 1k
        print('No section names found -> returning first 1k tokens')
        return ' '.join(row['sc-article'].split()[:1000])
    else:
        context = '\n'.join(found_sections.values())
        print('Found section names {}. Length:{}'.format(' - '.join(list(found_sections.keys())), len(context.split())))
        return context


def evaluate_conv(convs, convs_summ, gt_prs):
    #ROUGE SCORES
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(conv, gt_pr) for conv, gt_pr in zip(convs, gt_prs)]

    #BERT SCORE
    bert_scores = bertscore.compute(predictions=convs, references=gt_prs, lang="en")

    #DISTINCT-N over the questions
    #journalistic_questions = [turn['text'].split() for conv in parsed_convs for turn in conv if turn['author'] == 'Journalist']
    #print(journalistic_questions[:5])
    #q_dis_1, q_dis_2 = distinct_n_corpus_level(journalistic_questions, 1), distinct_n_corpus_level(journalistic_questions, 2)

    
    return {
        'rouge-1' : round(np.mean([s['rouge1'].fmeasure for s in rouge_scores]), 3),
        'rouge-L' : round(np.mean([s['rougeL'].fmeasure for s in rouge_scores]), 3),
        'bert-f1' : round(np.mean(bert_scores['f1']), 3),
        #'ques-distinct-1' : round(q_dis_1, 3),
        #'ques-distinct-2' : round(q_dis_2, 3),
        'rouge-scores': rouge_scores,
        'bert-scores': bert_scores
    }

def evaluate_text_similarity(pr_articles, gt_articles):
    #ROUGE SCORES
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(conv, gt_pr) for conv, gt_pr in zip(pr_articles, gt_articles)]

    # print(pr_articles[:10])
    # print(gt_articles[:10])
    # print(len(pr_articles))
    #BERT SCORE
    bert_scores = bertscore.compute(predictions=pr_articles, references=gt_articles, lang="en")

    
    return {
        'rouge-1' : round(np.mean([s['rouge1'].fmeasure for s in rouge_scores]), 3),
        'rouge-L' : round(np.mean([s['rougeL'].fmeasure for s in rouge_scores]), 3),
        'bert-f1' : round(np.mean(bert_scores['f1']), 3),
        'rouge-scores': rouge_scores,
        'bert-scores': bert_scores
    }

def parse_conversation(row):

    raw_conv = row['conversation']
    if raw_conv is None:
        print('None conversation')
        return {
            'parsed_conv': []
        }

    mapping_authors = {
        "J": "Journalist",
        "R": "Researcher",
        "Journalist": "Journalist",
        "Researcher":"Researcher",
        "Journalist (J)": "Journalist",
        "Researcher (R)": "Researcher"
    }

    try:
        # Remove any </think> tags
        if '</think>' in raw_conv:
            raw_conv = raw_conv.split('</think>')[-1]
        
        dlg = [utter.split(':') for utter in raw_conv.split('\n\n') if any(['**Journalist' in utter, '**Researcher' in utter, "**J" in utter, "**R" in utter])]
        dlg = [utter for utter in dlg if len(utter) > 1]
        dlg = [{'text': utter[1].replace("**", ""), 'author': utter[0].replace("**", "")} for utter in dlg]
        dlg = [{'text': utter["text"], 'author': re.sub(r"Researcher .*", "Researcher", utter["author"])} for utter in dlg]
        dlg = [{'text': utter["text"], 'author': mapping_authors[utter["author"]]} for utter in dlg]
        return {
            'parsed_conv': dlg
        }
    except:
        print('Error parsing')
        print(raw_conv)
        print('============')
        return {
            'parsed_conv': []
        }


def construct_full_dialogue(dataset, journalist_pipeline, researcher_pipeline, paper_title_clm='paper_title', paper_text_clm='paper_text', max_rounds=5, max_input_tokens=1500, max_journalist_turn_tokens=200, max_researcher_turn_tokens=500, journalist_prompt="You are a helpful and knowledgeable journalist asking questions about a scientific paper.", researcher_prompt = "You are a helpful and expert researcher answering questions about your scientific paper."):

    terminators = [
        journalist_pipeline.tokenizer.eos_token_id,
    ]
    
    def generate_response(pipe, messages, batch_size=1, max_new_tokens=100):
        prompts = [pipe.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        all_responses = []
        #print(prompts[0])
        #print('============================')
        responses = pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id= terminators,
            temperature=0.7,
            top_p=0.9,
            min_new_tokens=10,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        responses = [r[0]['generated_text'][len(prompts[i]):].strip() for i, r in enumerate(responses)]
        responses = [extract_complete_paragraphs(r) for r in responses]

        return responses

    # Take the first message about the system
    # We maintain two lists to address the role alteration
    journalist_generated_conversations = [[{"content": journalist_prompt, "role": "system"}] + [{'role': 'user', 'content': "[PAPERT-TITLE]\n{}\n[PAPER]\n{}".format(row[paper_title_clm], truncate_text(journalist_pipeline.tokenizer, row[paper_text_clm], max_input_tokens))}] for row in dataset]
    researcher_generated_conversations = [[{"content": researcher_prompt, "role": "system"}] + [{'role': 'user', 'content': "[PAPERT-TITLE]\n{}\n[PAPER]\n{}".format(row[paper_title_clm], truncate_text(journalist_pipeline.tokenizer, row[paper_text_clm], max_input_tokens))}] for row in dataset]
 
    for i in tqdm(range(max_rounds)):
        #journalist response
        responses = generate_response(journalist_pipeline, journalist_generated_conversations, max_new_tokens=max_journalist_turn_tokens)
        #print(responses)
        journalist_generated_conversations = [conv[0] + [{'content': conv[1], "role":"assistant"}] for conv in zip(journalist_generated_conversations, responses)]
        researcher_generated_conversations = [conv[0] + [{'content': conv[1], "role":"user"}] for conv in zip(researcher_generated_conversations, responses)]

        #Researcher response
        responses = generate_response(researcher_pipeline, researcher_generated_conversations, max_new_tokens=max_researcher_turn_tokens)
        #print(responses)
        journalist_generated_conversations = [conv[0] + [{'content': conv[1], "role":"user"}] for conv in zip(journalist_generated_conversations, responses)]
        researcher_generated_conversations = [conv[0] + [{'content': conv[1], "role":"assistant"}] for conv in zip(researcher_generated_conversations, responses)]

    dataset = dataset.add_column('generated_conversation', journalist_generated_conversations)
    dataset = dataset.map(lambda row: {'conversation': '\n\n'.join(['{}: {}'.format('Journalist', x['content'].replace('Researcher: ', '').replace('Journalist: ', '')) if x['role'] == 'assistant' else '{}: {}'.format('Researcher', x['content']) for x in row['generated_conversation'][2:]])})
    
    return dataset

if __name__ == "__main__":
    #base_model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    #adapter_name = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/llama3-trained-researcher-on-deepseek/'
    #output_path = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/llama3-trained-researcher-on-deepseek-full/'
    
    base_model_path = 'Qwen/Qwen2.5-7B-Instruct'
    adapter_name = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/qwen-trained-journalist-on-deepseek/'
    output_path = '/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/qwen-trained-journalist-on-deepseek-full/'
    
    model, tokenizer = load_model_with_adapter(base_model_path, adapter_name, device_map="cuda:0")
    model = model.merge_and_unload()
    model.save_pretrained(output_path)