import json
import ast
import torch
import pandas as pd
from typing import Literal, Any, Tuple, Union, List, Dict
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel


MEMOTRAP_DATAPATH = '/rds/user/ssc42/hpc-work/memotrap-testing/memotrap/1-proverb-ending.csv'
PROMPTS_PATH = '../nq_prompts.json'
llm_name = "huggyllama/llama-7b"
k = 10  # number of top tokens to consider


df = pd.read_csv(MEMOTRAP_DATAPATH)
with open(PROMPTS_PATH, 'r') as file:
    prompts: List[str] = json.load(file)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(llm_name)
model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path=llm_name,
    torch_dtype=torch.float16,
    device_map='auto',
)


n_questions, n_prompts = len(df), len(prompts)
result_ids = torch.zeros((n_questions, n_prompts, k), dtype=torch.int16)
result_probs = torch.zeros((n_questions, n_prompts, k), dtype=torch.float16)
print(device)


for question_idx, row in df.iterrows():
    context: str = row['prompt'].split(":")[0]
    question: str = row['prompt'].split(":")[1][1:]
    classes: List[str] = ast.literal_eval((row['classes']))
    answer_index: Literal[0, 1] = row['answer_index']

    for prompt_idx, prompt in enumerate(prompts):
        print("Prompt no.", prompt_idx)
        prompt = prompt.format(context=context, question=question)
        inputs: Any = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
            max_new_tokens=1,
            min_new_tokens=1,
        )
        distrib = outputs.scores[0].flatten()  # shape: (1, 32000)
        top_k_values, top_k_indices = torch.topk(distrib, k)  # shape: (k,)
        top_k_probs = torch.softmax(top_k_values, dim=0)  # shape: (k,)

        result_ids[question_idx, prompt_idx, :] = top_k_indices
        result_probs[question_idx, prompt_idx, :] = top_k_probs

print(result_ids)
print(result_probs)
torch.save(result_ids, 'result_ids.pt')
torch.save(result_probs, 'result_probs.pt')
