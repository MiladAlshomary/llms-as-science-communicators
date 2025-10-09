### Synthesizing conversation dataset


### Processing conversation dataset for SFT training

For LLAMA-3:

python prepare_dataset.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-cleaned/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-preprocessed-for-llama3/ meta-llama/Meta-Llama-3-8B-Instruct --max_num_turns=40

For Qwen:

prepare_dataset.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-cleaned/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-preprocessed-for-qwen/ Qwen/Qwen2.5-7B-Instruct --max_num_turns=40


### Training

CUDA_VISIBLE_DEVICES=0,1 python /local/nlp/milad/code/llms-as-science-communicators/src-py/training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-deepseek-final-conv-ds-preprocessed-for-llama3/train_journalist_ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 3 --run_name llama3-trained-journalist --model_name meta-llama/Meta-Llama-3-8B-Instruct --eval_steps 100 --lora_rank 16 --lora_alpha 32 --batch_size 16 --gradient_accumulation_steps 16






### Generating Conversations

CUDA_VISIBLE_DEVICES=1,5 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/baseline_qwen_gen_conv --journalist_base_model_name Qwen/Qwen2.5-7B-Instruct --researcher_base_model_name Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=2,6 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/baseline_llama3_gen_conv --journalist_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --researcher_base_model_name meta-llama/Meta-Llama-3-8B-Instruct

CUDA_VISIBLE_DEVICES=4,7 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/ft_llama3_gen_conv --journalist_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --journalist_adapter_name /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-llama3-trained-journalist-on-deepseek/  --researcher_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --researcher_adapter_name /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-llama3-trained-researcher-on-deepseek/

CUDA_VISIBLE_DEVICES=1,5 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/ft_qwen_gen_conv --journalist_base_model_name Qwen/Qwen2.5-7B-Instruct --journalist_adapter_name /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-qwen-trained-journalist-on-deepseek-3epochs/  --researcher_base_model_name Qwen/Qwen2.5-7
B-Instruct --researcher_adapter_name /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-qwen-trained-researcher-on-deepseek-3epochs/


### Generating Conversations with fixed researcher

### Qwen baseline
CUDA_VISIBLE_DEVICES=0,1 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/baseline_qwen_gen_conv_fixed_researcher --researcher_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --journalist_base_model_name Qwen/Qwen2.5-7B-Instruct --baseline_researcher

### Qwen fine-tuned
CUDA_VISIBLE_DEVICES=3,6 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/ft_qwen_gen_conv_fixed_researcher --researcher_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --journalist_base_model_name Qwen/Qwen2.5-7B-Instruct --journalist_adapter_name /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-qwen-trained-journalist-on-deepseek-3epochs/ --baseline_researcher

### llama3 baseline
CUDA_VISIBLE_DEVICES=0,1 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/ft_llama3_gen_conv_fixed_researcher --researcher_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --journalist_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --baseline_researcher

### llama3 fine-tuned
CUDA_VISIBLE_DEVICES=3,6 python generate_conversations.py /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/processed_test_ds_sample/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/new-eval_experiment_500/ft_llama3_gen_conv_fixed_researcher --researcher_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --journalist_base_model_name meta-llama/Meta-Llama-3-8B-Instruct --journalist_adapter_name /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/new-llama3-trained-journalist-on-deepseek-3epochs/ --baseline_researcher


### Running VLLM server

Running the vllm server:

#### Finte-tuned Llama-3 journalist
CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 3000 --enable-lora --lora-modules 'llm_journalist'="/mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/llama3-trained-journalist-on-deepseek/" --port 7777

#### Baseline Llama-3 journalist
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len 3000 --port 7789

#### Baseline DeepSeek journalist
CUDA_VISIBLE_DEVICES=4 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max-model-len 3000 --port 7790