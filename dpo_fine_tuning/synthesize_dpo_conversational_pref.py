
from openai import OpenAI
from pydantic import BaseModel, Field
from openai.lib._pydantic import to_strict_json_schema
from tqdm import tqdm
from enum import Enum
from typing import List, Dict, Any, Optional

import os
import json
import datasets
import argparse
from tqdm import tqdm

from huggingface_hub import login

login(os.environ['hf_token'])

class BinaryDecision(int, Enum):
    """Binary decision for Yes/No."""
    Yes  = 1
    No   = 0    

class EvalSchema(BaseModel):
    """Schema to evaluate if an answer is vague or contains complex aspects."""
    is_vague: BinaryDecision = Field(..., description="Is the answer vague or too generic?")
    complex_aspects: List[str] = Field(..., description="List of concepts that are highly technical that only an expert in the filed would understand. The list can be empty if the answer is not complex.")

eval_client = OpenAI(base_url="http://localhost:9988/v1", api_key="-")
eval_client2 = OpenAI(base_url="http://localhost:7777/v1", api_key="-")

def rate_answer_quality(dataset, evaluator_name, eval_prompt, input_name):
    eval_scores = []
    for row in tqdm(dataset):
        instruction = '{}\n\n{}'.format(eval_prompt, "[TEXT]: {}".format(row[input_name]))
        #print(instruction)
        completion = eval_client.chat.completions.create(
            model= evaluator_name,
            messages=[
                {
                    "role": "user",
                    "content": instruction,
                },
            ],
            response_format={"type": "json_schema", "json_schema": {"name":"eval_scheme", "schema": to_strict_json_schema(EvalSchema)}},
            temperature=0.0,
        )
        print(completion.choices[0].message)
        # The JSON response is in the `content` attribute
        score = json.loads(completion.choices[0].message.content)
        eval_scores.append(score)
        
    dataset = dataset.add_column('scoring_parsed', eval_scores)
    return dataset

def generate_questions(paper: str, conversation_history: List[Dict[str, str]], last_answer_eval: Dict[str, Any], generator_model: str) -> Dict[str, str]:
    """
    Generates a preferred (follow-up) and a rejected (generic) question based on the last answer's evaluation.

    Args:
        paper: The scientific paper text.
        conversation_history: The history of the conversation.
        last_answer_eval: The evaluation of the last researcher's answer.
        generator_model: The name of the model to use for generation.

    Returns:
        A dictionary containing the 'chosen' (preferred) and 'rejected' questions.
    """
    # Construct a prompt to generate a good follow-up question
    if last_answer_eval['is_vague'] == BinaryDecision.Yes:
        follow_up_instruction = "You are a smart research journalist. Based on the paper and conversation, the last answer from the researcher was evaluated as follows:\n"
        follow_up_instruction += "- The answer was vague.\n"
        follow_up_instruction += "\nYour task is to ask a single, concise follow-up question to address these points, seeking clarification and more depth. Do not be conversational, just output the question."
    if last_answer_eval['complex_aspects']:
        follow_up_instruction = "You are a smart research journalist. Based on the paper and conversation, the last answer from the researcher was evaluated as follows:\n"
        follow_up_instruction += f"- The answer contained these complex aspects that need clarification: {', '.join(last_answer_eval['complex_aspects'])}.\n"
        follow_up_instruction += "\nYour task is to ask a single, concise follow-up question to address these points, seeking clarification and more depth. Do not be conversational, just output the question."
    else:
        follow_up_instruction = "You are a smart research journalist. The last answer was clear. Now, ask a question about the societal impact of the research paper. Do not be conversational, just output the question."

    # Construct a prompt to generate a generic, less-optimal question
    generic_instruction = "You are a research journalist. Based on the paper and conversation, ask a new, generic question about the research. The question should not be a direct follow-up to the last answer. Do not be conversational, just output the question."

    content_prompt = f"[PAPER]:\n{paper}\n\n[CONVERSATION HISTORY]:\n{json.dumps(conversation_history, indent=2)}"

    #print(content_prompt)

    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': content_prompt}
    ]

    # Generate preferred question
    messages[0]['content'] = follow_up_instruction
    completion_chosen = eval_client2.chat.completions.create(
        model=generator_model,
        messages=messages,
        temperature=0.7,
        top_p=0.9,
    )
    chosen_question = completion_chosen.choices[0].message.content.strip()

    # Generate rejected question
    messages[0]['content'] = generic_instruction
    completion_rejected = eval_client2.chat.completions.create(
        model=generator_model,
        messages=messages,
        temperature=0.7,
        top_p=0.9,
    )
    rejected_question = completion_rejected.choices[0].message.content.strip()

    return {"prompt": content_prompt, "chosen": chosen_question, "rejected": rejected_question}


def generate_journalist_preference_data(row: Dict[str, Any], evaluator_model: str, generator_model: str) -> Optional[Dict[str, Any]]:
    """
    This will generate preference data for a given paper and its conversation_history.
    The code will test if the last utterance in the conversation history is vague or complex, and if so,
    it will generate two questions: one preferred (follow-up) and one rejected (generic).

    Args:
        paper: The scientific paper text.
        row: A dictionary containing the data for a single sample.
        generator_model: The model to generate the questions.

    Returns:
        A dictionary containing the prompt, chosen, and rejected questions for DPO, or None if no preference data is generated.
    """

    #print(paper)
    paper = row['prompt'][1]['content'].split('[PAPER]')[-1]
    conversation_history = row['prompt'][2:]

    #print(conversation_history[-1])

    #print(conversation_history)

    if not conversation_history or conversation_history[-1]['role'] != 'user': # Assuming 'user' is the researcher
        print("Last utterance is not from the researcher. Skipping.")
        return {
            "chosen": None,
            "rejected": None,
            "answer_quality": None
        }

    if 'is_vague' in row and 'complex_aspects' in row:
        print("Found existing evaluation fields. Skipping evaluation.")
        last_answer_eval = {
            'is_vague': row['is_vague'],
            'complex_aspects': row['complex_aspects']
        }
    else:
        try:
            last_answer = conversation_history[-1]['content']
        
            # 1. Evaluate the last answer for vagueness and complexity
            eval_prompt = """
            Please evaluate the following text from a researcher. 
            Identify the following:
             1. If the answer is vague and omitting important details.
             2. List of concepts that are highly technical that only an expert in the filed would understand. The list can be empty if the answer is not complex.
            """
            instruction = f"{eval_prompt}\n\n[TEXT]: {last_answer}"
        
            completion = eval_client.chat.completions.create(
                model=evaluator_model,
                messages=[{"role": "user", "content": instruction}],
                response_format={"type": "json_schema", "json_schema": {"name": "eval_scheme", "schema": to_strict_json_schema(EvalSchema)}},
                temperature=0.0,
            )
            last_answer_eval = json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f'Exception during evaluation, skipping: {e}')
            return { "chosen": None, "rejected": None, "answer_quality": None }

    # 2. If the answer is vague or has complex parts, generate preference pair
    if last_answer_eval['is_vague'] == BinaryDecision.Yes or last_answer_eval['complex_aspects']:
        print("Last answer is vague or complex. Generating preference pair for clarification...")
    else:
        print("Last answer is clear. Generating preference pair for societal impact.")

    result = generate_questions(paper, conversation_history, last_answer_eval, generator_model)
    return {
        "chosen": result['chosen'],
        "rejected": result['rejected'],
        "answer_quality": last_answer_eval
    }


def main(args):

    dataset = datasets.load_dataset(args.dataset_name, split=args.dataset_split).select(range(args.num_samples))
    dataset = dataset.map(lambda row: generate_journalist_preference_data(row, args.evaluator_model,
                                            args.generator_model))
    dataset = dataset.filter(lambda row: row['chosen'] is not None)
    dataset.save_to_disk(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO preference data for journalist questions.")
    parser.add_argument("--dataset_name", type=str, default='miladalsh/new-deepseek-final-conv-ds-cleaned-and-processed',
                        help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_split", type=str, default='train',
                        help="Dataset split to use.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to process from the dataset.")
    parser.add_argument("--evaluator_model", type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                        help="Model to use for evaluating answer quality.")
    parser.add_argument("--generator_model", type=str,
                        default='/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-llama3-trained-journalist-on-deepseek-3epochs-final-full-model/',
                        help="Model to use for generating questions.")
    parser.add_argument("--output_dir", type=str, default='dpo_ds',
                        help="Directory to save the generated dataset.")
    args = parser.parse_args()
    main(args)