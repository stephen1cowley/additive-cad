from typing import List, Literal
import re
import string
import ast
import torch
import pandas as pd
from transformers import LlamaTokenizer, PreTrainedTokenizer


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def eval_token_validity(
    tokenizer: PreTrainedTokenizer,
    token_id: int,
    answer: str
) -> bool:
    "Returns `True` if the token is a valid first token for the answer"
    token = normalize_answer(tokenizer.decode(token_id))
    if len(token) == 0:
        return False
    return token == normalize_answer(answer)[0:len(token)]


llm_name = "huggyllama/llama-7b"
tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(llm_name)

MEMOTRAP_DATAPATH = '1-proverb-ending.csv'
df = pd.read_csv(MEMOTRAP_DATAPATH)

result_ids = torch.load('result_ids.pt')
result_probs = torch.load('result_probs.pt')

print(result_ids.shape)
print(result_probs.shape)

num_q = result_ids.shape[0]
num_prompts = result_ids.shape[1]
k = result_ids.shape[2]
solution = torch.zeros(num_prompts)


for q_num in range(num_q):
    p_num = 0

    context: str = df['prompt'][q_num].split(":")[0]
    question: str = df['prompt'][q_num].split(":")[1][1:]
    classes: List[str] = ast.literal_eval((df['classes'][q_num]))
    answer_index: Literal[0, 1] = df['answer_index'][q_num]

    # print(f"{context} {question} {classes[answer_index]}")

    best_prob = 0
    best_token = torch.tensor([])
    for prompt_id in range(num_prompts):
        for k_id in range(k):
            corr_ans = eval_token_validity(
                tokenizer,
                result_ids[q_num, prompt_id, k_id],
                classes[answer_index]
            )
            if corr_ans and result_probs[q_num, prompt_id, k_id] > best_prob:
                best_prob = result_probs[q_num, prompt_id, k_id]
                best_token = result_ids[q_num, prompt_id, k_id]
            #     print("\nBest token is", repr(tokenizer.decode(best_token)), "\n")

            # print(repr(tokenizer.decode(result_ids[q_num, prompt_id, k_id])),
            #     f"{result_probs[q_num, prompt_id, k_id].item():.4f}",
            #     corr_ans)
    # Reshape the indices and values for sparse tensor creation
    if best_prob == 0:  # No valid token found
        continue
    indices = torch.stack([
        result_ids[q_num, :, :].reshape(-1),          # Row indices (token ids)
        torch.repeat_interleave(torch.arange(num_prompts), k)  # Column indices (prompts)
    ])
    values = result_probs[q_num, :, :].reshape(-1)

    sparse_probs = torch.sparse_coo_tensor(
        indices=indices,
        values=values.to(torch.float32),
        size=torch.Size([32000, num_prompts])
    )
    # print(sparse_probs.shape)
    # print(sparse_probs)

    sparse_correct = torch.sparse_coo_tensor(best_token.unsqueeze(0).unsqueeze(0),
                                             torch.tensor([1.0]),
                                             torch.Size([32000]))
    # Convert sparse tensors to dense before operations
    # sparse_probs_dense = sparse_probs.to_dense().to(torch.float32)
    # sparse_correct_dense = sparse_correct.to_dense()
    # print(sparse_correct_dense.shape)

    solution = solution + sparse_probs.to_dense()[best_token, :]

tikhonov_reg = 0.001
print(solution)

# TODO: evaluate the accuracy at correct first token of this particular combination.
