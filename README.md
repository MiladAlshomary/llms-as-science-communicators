### Usefull Command for training:

#### Training LLAMA-3 model

**Researcher**: 
CUDA_VISIBLE_DEVICES=3 python training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-scienc
e-to-the-public/deepseek-final-conv-ds-preprocessed-for-llama3/train_researcher_ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 1 --run_name llama3-trained-researcher-on-deepseek --eval_steps 100 --lora_rank 16 --lora_alpha 32

**Journalist**: 
CUDA_VISIBLE_DEVICES=1 python training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-preprocessed-for-llama3/train_journalist_ds /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 1 --run_name llama3-trained-journalist-on-deepseek --eval_steps 100 --lora_rank 16 --lora_alpha 32


#### Training Qwen model

**Researcher**:
/local/nlp/milad/code/communicating-science-to-the-public/training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-preprocessed-for-qwen/train_researcher_ds/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 3 --run_name qwen-trained-researcher-on-deepseek-for-40k-samples --model_name Qwen/Qwen2.5-7B-Instruct --eval_steps 50 --lora_rank 16 --lora_alpha 32 --batch_size=1 --gradient_accumulation_steps 16 --gradient_accumulation_steps 16

**Journalist**:
/local/nlp/milad/code/communicating-science-to-the-public/training_llm_on_conversations.py train /mnt/swordfish-pool2/milad/communicating-science-to-the-public/deepseek-final-conv-ds-preprocessed-for-qwen/train_journalist_ds/ /mnt/swordfish-pool2/milad/communicating-science-to-the-public/models/ --training_epochs 3 --run_name qwen-trained-journalist-on-deepseek-for-40k-samples --model_name Qwen/Qwen2.5-7B-Instruct --eval_steps 50 --lora_rank 16 --lora_alpha 32 --batch_size=1 --gradient_accumulation_steps 16 --gradient_accumulation_steps 16
