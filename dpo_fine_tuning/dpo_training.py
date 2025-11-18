import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import torch
import wandb

def tokenize_dpo_pair(example, tokenizer):
    """
    Tokenizes a DPO pair. The prompt is a list of dicts representing a conversation.
    The chosen and rejected responses are strings.
    """
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Apply chat template to the prompt, which is a conversation history
    prompt_tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)

    return {
      "prompt": torch.tensor(prompt_tokens),
      "chosen": tokenizer(chosen, add_special_tokens=False).input_ids,
      "rejected": tokenizer(rejected, add_special_tokens=False).input_ids,
    }

def train_model(args):
    wandb.init(project="training-llama-on-conversations", name=args.run_name)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare model for k-bit + LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    #ds = load_dataset(args.dataset_path)
    ds = load_from_disk(args.dataset_path)
    ds = ds.remove_columns(['paper_id', 'paper_title', 'paper_text', 'completion', 'answer_quality'])
    ds = ds.map(lambda x: {'prompt': tokenizer.apply_chat_template(x['prompt'], tokenize=False, add_generation_prompt=True)})

    #split ds into training and validation
    ds = ds.train_test_split(test_size=0.1, seed=42)
    
    # DPO config & trainer
    dpo_config = DPOConfig(
        run_name=args.run_name,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        eval_strategy="steps",
        beta=args.beta,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # No separate reference model
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        processing_class=tokenizer,
        peft_config=lora_config,
        args=dpo_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DPO model.")
    parser.add_argument("--model_name", type=str, required=True, help="SFT checkpoint to use for DPO training.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer to use.")
    parser.add_argument("--dataset_path", type=str, default="pairs.jsonl", help="Path to the training data in JSONL format.")
    parser.add_argument("--run_name", type=str, default="dpo_run", help="A name for the wandb run.")
    parser.add_argument("--output_dir", type=str, default="dpo-journalist-7b-lora", help="Directory to save the trained model.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--max_length", type=int, default=2000, help="Maximum sequence length.")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "v_proj"], help="Modules to apply LoRA to.")

    args = parser.parse_args()
    train_model(args)