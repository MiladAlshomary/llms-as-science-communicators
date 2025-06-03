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



llm_to_use='chat-gpt'

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


def load_model_with_adapter(base_model_path, adapter_path="", device_map="auto"):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    if adapter_path != "":
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model.load_adapter(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token  # set pad token
        tokenizer.padding_side = "right"
        return base_model, tokenizer
    else:
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
    
def generate_llm_response(prompt):
    if llm_to_use=='chat-gpt':
        output = client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        return output

    if llm_to_use=='llama3':
        outputs = llama3_pipeline(
            text_inputs=[{"role": "user", "content": prompt}],
            #max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][-1]["content"]
        
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

#Generating Focus Questions from a Given Context
def generate_questions(pr_article):
    prompt = focus_area_prompt.replace("[pr-article]", pr_article)
    output = generate_llm_response(prompt)#client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    return output

#Tagging PR for sentence level scientific rhetorical role
def tag_pr_article(pr_article):
    prompt = gpt_assign_label.replace("[pr_article]", pr_article)
    output =  generate_llm_response(prompt)#client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    return output

# functions for deep dialogue generations for LLM Journalist and LLM Researcher Conversations
def parse_gpt_feedback(output):
    question = output.split("NEXT QUESTION:")
    return question[1].strip()

def select_sentences_from_pr(desired_label, parsed_pr):

    select_sentences = []

    if isinstance(parsed_pr, str):
        parsed_pr = ast.literal_eval(parsed_pr)

    for element in parsed_pr:
        if element["Label"] == desired_label: 
            select_sentences.append(element["Sentence"])

    print(select_sentences)
    return "\n".join(select_sentences)

def split_dialogue_to_json(dialogue):
    # Regular expression to match each speaker's utterance
    pattern = r"(Reporter|Researcher):"
    
    # Split the dialogue based on the pattern
    parts = re.split(pattern, dialogue)
    
    # Remove any leading or trailing whitespace from each part
    parts = [part.strip() for part in parts if part.strip()]
    
    # Create a list of dictionaries
    utterances = []
    for i in range(0, len(parts), 2):
        speaker = parts[i]
        utterance = parts[i+1]
        utterances.append({"author": speaker, "text": utterance})
    
    return utterances

# LLM Journalist/Researcher Prompts + Outputs
def journalist_first_utterance(journalist_initial_prompt, focus_question, pr_article, sc_title):
    prompt = journalist_initial_prompt.replace("[focus question]", focus_question).replace("[pr-article]", pr_article).replace("[sc_title]", sc_title)
    output =  generate_llm_response(prompt)#client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    return output

def generate_response_journalist(conversation, prompt, num_turns, focus_question, parsed_pr, next_question):
    if num_turns == 20:
        return "<STOP>"
    else:
        prompt = journalist_prompt_with_feedback.replace("[focus question]", focus_question).replace("[parsed_pr]", parsed_pr).replace("[num turns]", str(num_turns)).replace("[conversation]", conversation).replace("[next_question]", next_question)
        output =  generate_llm_response(prompt)#client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

        return output
    
def generate_response_researcher(conversation, prompt, num_turns, title, abstract, introduction):
    if num_turns == 20:
        return "<STOP>"
    else:
        prompt = prompt.replace("[num turns]", str(num_turns)).replace("[conversation]", conversation).replace("[abstract]", abstract).replace("[title]", title).replace("[introduction]", introduction)
        output =  generate_llm_response(prompt)#client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        return output

# Generating Evaluation Criteria
def eval_convo(conversation, focus_question, pr_article):
    prompt = llm_journalist_evaluator.replace("[conversation]", conversation).replace("[focus area]", focus_question).replace("[pr-article]", pr_article)
    output =  generate_llm_response(prompt)#client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    
    return output


def generate_dialogue(questions_from_pr_summary, questions_from_pr_article, pr_summary, pr_article, pr_summary_classes, pr_article_classes, sc_title, sc_abstract, sc_introduction):

    num_turns = 0

    conversation_tracker = {}

    if isinstance(questions_from_pr_article, str):
        questions_from_pr_article = ast.literal_eval(questions_from_pr_article)

    for question in questions_from_pr_article:

        focus_question = question["Focus Question"]
        focus_label = question["Label"]

        parsed_pr = select_sentences_from_pr(focus_label, pr_article_classes)
        parsed_pr += select_sentences_from_pr(focus_label, pr_summary_classes)

        print("parsed_pr:", parsed_pr)

        print("FOCUS QUESTION: ", focus_question)

        #initial journalist prompt
        output = journalist_first_utterance(journalist_initial_prompt, focus_question, pr_article, sc_title)
        if "Journalist" not in output:
            conversation_utterances = f"Journalist: {output}"
        else:
            conversation_utterances = output
        
        print(conversation_utterances)
    
        current_role = "researcher"
        num_turns = 0

        next_question = ""

        #continue generating dialogue until <STOP> is generated 
        while "<STOP>" not in output or "<STOP>" not in result:
            if current_role == "researcher":
                output = generate_response_researcher(conversation_utterances, researcher_prompt, num_turns, sc_title, sc_abstract, sc_introduction)

                if "Researcher: " not in output:
                    conversation_utterances += f"\nResearcher: {output}"
                else:
                    conversation_utterances += f"\n{output}"

                print("conversation_utterances: ", conversation_utterances)

                #only evaluate after the researcher has responded. and skip the first evaluation
                if num_turns != 0:
                    result = eval_convo(conversation_utterances, focus_question, pr_article)
                    next_question = parse_gpt_feedback(result)
                    print("Next Question: ", next_question)
                    
                current_role = "journalist"
                
            else:
                output = generate_response_journalist(conversation_utterances, journalist_prompt_with_feedback, num_turns, focus_question, parsed_pr, next_question)

                if "Journalist: " not in output:
                    conversation_utterances += f"\nJournalist: {output}"
                else:
                    conversation_utterances += f"\n{output}"
                
                # print("conversation_utterances: ", conversation_utterances)

                current_role = "researcher"
            num_turns +=1
        
        # turns = split_dialogue_to_json(conversation_utterances)
        conversation_tracker[focus_question] = {"utterances": conversation_utterances}
        
        
    return conversation_tracker    
    # return conversation_utterances, turns

def build_model_context_grace(sections, section_names):
    context = ["introduction"]
    
    if isinstance(section_names, str):
        section_names = ast.literal_eval(section_names)
    
    section_names = [x.lower() for x in section_names]
    print(section_names)
    
    context_to_idx = {}
    for c in context:
        for i, s in enumerate(section_names):
            if c in s:
                print("hi")
                context_to_idx[c] = i
                print(context_to_idx)

    if isinstance(sections, str):
        sections = ast.literal_eval(sections)

    found_sections = {s[0]: sections[s[1]] for s in context_to_idx.items()}
    return found_sections

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

def complete_dialogue(row, pipe, batch_size=1, max_new_tokens=500):
    terminators = [
        pipe.tokenizer.eos_token_id,
    ]
    
    messages = row['messages']
    prompts  = []
    for i in range(2, len(messages), 2):
        prompt = pipe.tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    outputs = []
    for prompts_batch in np.array_split(prompts, len(prompts)/batch_size):
        #print(prompts_batch.tolist())
        responses = pipe(
            prompts_batch.tolist(),
            max_new_tokens=max_new_tokens,
            eos_token_id= terminators,
            do_sample=True,
            temperature=0.1,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        #responses = pipe(prompts_batch.tolist(), max_new_tokens=500, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
        responses = [r[0]['generated_text'][len(prompts_batch.tolist()[i]):].strip() for i, r in enumerate(responses)]
        outputs+= responses

    return outputs

def complete_dialogues(model_name, dataset, output_clm, adapter_name=''):
    # Load our fine-tuned llama-3 model Model with PEFT adapter
    llama3_model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",
      torch_dtype=torch.float16
    )

    if adapter_name != '':
        lora_config = LoraConfig(
            target_modules=["q_proj", "k_proj"],
            init_lora_weights=False
        )
        
        llama3_model.add_adapter(lora_config, adapter_name=adapter_name)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    llama3_pipe = pipeline("text-generation", model=llama3_model, tokenizer=tokenizer, batch_size=8)
    dataset  = dataset.map(lambda row: {output_clm: complete_dialogue(row, llama3_pipe)})
    return dataset

def construct_full_dialogue(dataset, journalist_pipeline, researcher_pipeline, max_rounds=5, max_journalist_turn_tokens=200, max_researcher_turn_tokens=200):

    journalist_prompt = """You are a helpful and knowledgeable journalist asking questions about a scientific paper."""
    researcher_prompt = """You are a helpful and expert researcher answering questions about your scientific paper."""

    terminators = [
        journalist_pipeline.tokenizer.eos_token_id,
    ]
    
    def generate_response(pipe, messages, max_new_tokens, batch_size=1):
        prompts = [pipe.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        
        #print(prompts[0])
        all_responses = []
        responses = pipe(
            prompts,
            # max_length=max_input_tokens,
            # truncation= True,
            max_new_tokens=max_new_tokens,
            eos_token_id= terminators,
            do_sample=True,
            temperature= 0.5,
            top_p=0.5,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        #print(responses[0][0]['generated_text'])
        responses = [r[0]['generated_text'][len(prompts[i]):].strip() for i, r in enumerate(responses)]

        return responses

    
    # Take the first message about the system
    # We maintain two lists to address the role alteration
    journalist_generated_conversations = [[{"content": journalist_prompt , "role": "system"}] for row in dataset]
    researcher_generated_conversations = [[{"content": researcher_prompt, "role": "system"}] for row in dataset]

    so_far_conversations = [[{'author':'Journalist', 'text': 'What is this paper about?'}] for item in dataset]
    for i in range(max_rounds):
        

        #Researcher response
        so_far_conversations_as_str = [" ".join(['{}: {}'.format(x['author'], x['text']) for x in so_far_conversation]).strip() for so_far_conversation in so_far_conversations]
        researcher_prompts = [[{"content": researcher_prompt , "role": "system"}] + [{"content" : "[PAPER]\n{}\n\n[CONTEXT]\n{}\n\n".format(row['sc-intro'], so_far_conversations_as_str[i]), "role": "user"}] for i, row in enumerate(dataset)]
        responses = generate_response(researcher_pipeline, researcher_prompts, max_researcher_turn_tokens)


        so_far_conversations = [conv[0] + [{'author': "Researcher" , "text":conv[1]}] for conv in zip(so_far_conversations, responses)]

        
         #journalist responses
        so_far_conversations_as_str = [" ".join(['{}: {}'.format(x['author'], x['text']) for x in so_far_conversation]).strip() for so_far_conversation in so_far_conversations]
        journalist_prompts = [[{"content": journalist_prompt , "role": "system"}] + [{"content" : "[PAPER]\n{}\n\n[CONTEXT]\n{}\n\n".format(row['sc-intro'], so_far_conversations_as_str[i]), "role": "user"}] for i, row in enumerate(dataset)]
        responses = generate_response(journalist_pipeline, journalist_prompts, max_journalist_turn_tokens)
        
        so_far_conversations = [conv[0] + [{'author': "Journalist" , "text":conv[1]}] for conv in zip(so_far_conversations, responses)]
        print('\n\n'.join(['[[{}]]: {}'.format(x['author'], x['text']) for x in so_far_conversations[0]]))
        print('\n\n\n====================')
        
    dataset = dataset.add_column('generated_conversation', journalist_generated_conversations)
    
    return dataset

def parse_conversation(row):
    raw_conv = row['conversation']
    if raw_conv is None:
        print('None conversation')
        return {
            'parsed_conv': []
        }

    # Remove any </think> tags
    if '</think>' in raw_conv:
        raw_conv = raw_conv.split('</think>')[-1]
    
    dlg = [utter.split(':') for utter in raw_conv.split('\n\n') if 'Journalist' in utter or 'Researcher' in utter]
    dlg = [utter for utter in dlg if len(utter) > 1]
    dlg = [{'text': utter[1].replace("**", ""), 'author': utter[0].replace("**", "")} for utter in dlg]
    return {
        'parsed_conv': dlg
    }

def construct_full_dialogue_method_3(dataset, journalist_pipeline, researcher_pipeline, max_rounds=5, max_input_tokens=1500, max_journalist_turn_tokens=200, max_researcher_turn_tokens=500):

    journalist_prompt = """ You are a great journalist that knows how to ask questions. You will converse with a researcher asking questions about their paper titled "{}"

      Paper:
      {}
      """
    researcher_prompt = """ You are a great researcher that can explain their research to the public. You published recently a paper titled "{}". 
      You will converse with a journalist answering their question about the paper to communicate your results and findings to the public

      Paper:
      {}
      """
    terminators = [
        journalist_pipeline.tokenizer.eos_token_id,
    ]
    
    def generate_response(pipe, messages, batch_size=1, max_new_tokens=100):
        prompts = [pipe.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        all_responses = []
        responses = pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id= terminators,
            do_sample=True,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        responses = [r[0]['generated_text'][len(prompts[i]):].strip() for i, r in enumerate(responses)]

        return responses

    # Take the first message about the system
    # We maintain two lists to address the role alteration
    journalist_generated_conversations = [[{"content": journalist_prompt.format(row['pr-title'], truncate_text(journalist_pipeline.tokenizer, row['sc-intro'], max_input_tokens)) , "role": "system"}] for row in dataset] #[m[:1] for m in dataset['messages']]
    researcher_generated_conversations = [[{"content": researcher_prompt.format(row['pr-title'], truncate_text(researcher_pipeline.tokenizer, row['sc-intro'], max_input_tokens)) , "role": "system"}] for row in dataset] #[m[:1] for m in dataset['messages']]

    #give the journalist a head start
    journalist_generated_conversations = [x + [{'role': 'user', 'content': 'Hi, please ask me whatever you like about the paper!'}] for x in journalist_generated_conversations]
    for i in range(max_rounds):
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
    
    return dataset

def construct_full_dialogue_method_4(dataset, journalist_pipeline, researcher_pipeline, paper_title_clm='paper_title', paper_text_clm='paper_text', max_rounds=5, max_input_tokens=1500, max_journalist_turn_tokens=200, max_researcher_turn_tokens=500, journalist_prompt="You are a helpful and knowledgeable journalist asking questions about a scientific paper.", researcher_prompt = "You are a helpful and expert researcher answering questions about your scientific paper."):

    terminators = [
        journalist_pipeline.tokenizer.eos_token_id,
    ]
    
    def generate_response(pipe, messages, batch_size=1, max_new_tokens=100):
        prompts = [pipe.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        all_responses = []
        # print(prompts[0])
        # print('============================')
        responses = pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id= terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            min_new_tokens=10,
            batch_size=batch_size,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        responses = [r[0]['generated_text'][len(prompts[i]):].strip() for i, r in enumerate(responses)]

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
    
def construct_full_dialogue_from_method_2(dataset, base_model_name, output_clm="conversation", max_input_tokens=1200, max_new_tokens=800, adapter_name=""):
    
    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # === Load LoRA adapter ===
    if adapter_name != "":
        base_model.load_adapter(adapter_name, adapter_name="default")
    
    base_model.eval()
    
    # === Set up pipeline ===
    llama3_pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    prompts = [
        [{"role": "system", "content": """You are an AI assistant helping a journalist writing an interview about a scientific paper."""},{
          "role": "user", "content": """ Here is a scientific paper:
                
                Title: {}
                Paper: {}
                
                Now, let's start the conversation.
                """.format(row['pr-title'], truncate_text(tokenizer, row['sc-intro'], max_input_tokens))
            }
        ] for row in dataset
    ]

    prompts = [llama3_pipe.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in prompts]
    #print(prompts)
    responses = llama3_pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id= [llama3_pipe.tokenizer.eos_token_id],
            do_sample=True,
            temperature= 0.5,
            top_p=0.5,
            pad_token_id=llama3_pipe.tokenizer.eos_token_id
        )
    responses = [r[0]['generated_text'][len(prompts[i]):].strip() for i, r in enumerate(responses)]

    dataset = dataset.add_column(output_clm, responses)

    return dataset