"""
The additive_cad module: includes the AdditiveCad class which allows simple llm inference with
original CAD as well as other decoding strategies.
"""

from typing import Literal, Any, Tuple, Union, List, Dict
import json
import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.generation.utils import ModelOutput
from src.experiment_types import ExperimentConfig
from src.utils import normalize_answer, evaluate_recall

class AdditiveCad:
    """
    Sets up an LLM that we can do inference on.
    """
    def __init__(self, config: ExperimentConfig):
        self.config: ExperimentConfig = config
        self.model_name: str = config.llm_name
        self.device: Literal['cpu', 'cuda'] = config.device
        self.tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )

    def generate_distribution(
            self,
            input_text: str,
            dola_layers: Literal['high', 'low', 'none'] = 'none',
        ) -> Union[torch.FloatTensor, None]:
        """
        Generate a single token either with DoLa, depending on whether `dola_layers` is set to the
        higher or lower layer setting. DoLa is turned off if `dola_layers=None`.
        Returns the logits of the token.
        """
        inputs: Any = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
            dola_layers=None if dola_layers == 'none' else dola_layers,
            max_new_tokens=1,
            min_new_tokens=1,
        )
        if not isinstance(outputs, ModelOutput):
            return None
        if isinstance(outputs.scores, Tuple):
            if isinstance(outputs.scores[0], torch.Tensor):
                return outputs.scores[0]
        return None

    def cad_decoding(
            self,
            bad_distribution: torch.Tensor,
            good_distribution: torch.Tensor,
            beta: float = 1.0,
        ) -> int:
        """
        Take 2 distributions, do contrastive decoding with adaptive plausibility constraint
        then return the token id with highest logit. Alpha and beta default to literature values.
        """
        apc = self.config.apc

        # Replace -inf with -1000 and inf with 1000
        # good and bad distributions are of shape (1, 32000)
        bad_distribution = torch.where(bad_distribution == float('-inf'), torch.tensor(-1000.0), bad_distribution)
        bad_distribution = torch.where(bad_distribution == float('inf'), torch.tensor(1000.0), bad_distribution)
        good_distribution = torch.where(good_distribution == float('-inf'), torch.tensor(-1000.0), good_distribution)
        good_distribution = torch.where(good_distribution == float('inf'), torch.tensor(1000.0), good_distribution)

        good_probs = torch.softmax(good_distribution, dim=-1)
        thresh = apc * float(torch.max(good_probs).item())
        plausible_ids = (good_probs > thresh).nonzero(as_tuple=True)[-1]

        max_logit = float('-inf')
        can_id = None

        # go through all plausible id logits, and find the maximum post-contrasting
        for i in plausible_ids:
            i = int(i)
            logit = (1 + beta) * good_distribution[0, i] - beta * bad_distribution[0, i]
            if logit > max_logit:
                max_logit = logit
                can_id = i
        if not can_id is None:
            return can_id
        return -1
    
    def add_cad_decoding(
            self,
            bad_distribution: torch.Tensor,
            good_distribution: torch.Tensor,
            gamma: float = 0.0,
        ) -> int:
        """
        Take 2 distributions, do contrastive decoding with adaptive plausibility constraint
        then return the token id with highest logit. Alpha and beta default to literature values.
        """
        # Replace -inf with -1000 and inf with 1000
        # good and bad distributions are of shape (1, 32000)
        bad_distribution = torch.where(bad_distribution == float('-inf'), torch.tensor(-1000.0), bad_distribution)
        bad_distribution = torch.where(bad_distribution == float('inf'), torch.tensor(1000.0), bad_distribution)
        good_distribution = torch.where(good_distribution == float('-inf'), torch.tensor(-1000.0), good_distribution)
        good_distribution = torch.where(good_distribution == float('inf'), torch.tensor(1000.0), good_distribution)

        good_probs = torch.softmax(good_distribution, dim=-1)
        bad_probs = torch.softmax(bad_distribution, dim=-1)

        new_probs = bad_probs + (good_probs - bad_probs) * (10**gamma)
        max_index = torch.argmax(new_probs).item()

        if not max_index is None:
            return int(max_index)
        return -1

    def generate_response(
            self,
            context: str,
            question: str,
            coeff: float
        ) -> Union[str, None]:
        """
        Return the generated response given a question and context, based on the settings of the
        object config.

        Args:
            context (str):
                The context to be placed in the `{context}` formatter of the prompts of the config file.
            question (str):
                The question to be placed in the `{question}` formatter of the prompts of the config file.
            coeff (float):
                The coefficient required for the configured contrastive decoding type.

        Returns:
            The LLM's response to a question. Returns `str` type if successful, or `None` if an error occured.
        """
        output: str = " "

        for token_number in range(self.config.max_tokens):
            good_distribution = self.generate_distribution(
                input_text=self.config.context_prompt.format(context=context, question=question) + output,
                dola_layers=self.config.dola_layers_context
            )
            bad_distribution = self.generate_distribution(
                input_text=self.config.no_context_prompt.format(question=question) + output,
                dola_layers=self.config.dola_layers_no_context
            )
            if good_distribution is not None and bad_distribution is not None:
                if self.config.decoding_strategy == 'CAD':
                    next_token_id = self.cad_decoding(
                        bad_distribution=bad_distribution,
                        good_distribution=good_distribution,
                        beta=coeff,
                    )
                elif self.config.decoding_strategy == 'additive-CAD':
                    next_token_id = self.add_cad_decoding(
                        bad_distribution=bad_distribution,
                        good_distribution=good_distribution,
                        gamma=coeff
                    )
                else:
                    return None
                if next_token_id == -1:
                    raise TypeError("cad_decoding failed to return correct id")
                
                output = self.tokenizer.decode(self.tokenizer.encode(output) + [next_token_id], skip_special_tokens=True)

                # stopping_symbols = [".", "\n"]
                # for stopping_symbol in stopping_symbols:
                #     if stopping_symbol in output:
                #         return output
            else:
                return None
        return output  # Assuming the space was taken out before context and prompt passed in

    def generate_results(self) -> Any:
        "Generate a set of results"

        max_time: float = 7*3600
        
        with open(self.config.dataset_path, 'r') as file:
            data: List[Any] = json.load(file)

        time_0 = time.time()

        results: Dict[str, int] = {}

        for coeff in self.config.test_coefficients:
            time_1 = time.time()

            score: int = 0
            for idx, qa in enumerate(data):
                context: str = qa["context"]
                question: str = qa["question"]
                answers: List[str] = qa["answer"]  # There may be multiple answers

                response = self.generate_response(
                    context=context,
                    question=question,
                    coeff=coeff
                )

                if response == None:
                    return None  # Error with CAD generation
                response = normalize_answer(response)

                if time.time() - time_0 >= max_time:
                    print("Time up!")
                    return score

                if idx % 100 == 0:
                    print(f"{idx}. CAD answer: {repr(response)}")
                    print(f"{idx}. Correct answers:", " ".join([repr(normalize_answer(answers[i])) for i in range(len(answers))]))
                    print("Time:", time.time() - time_0)
                if evaluate_recall(response, answers):
                    score += 1
            print(f"RESULT: CAD with coefficient={beta}, dola-good set to {dola_layers_good}, dola-bad set to {dola_layers_bad}, model {llm.model_name}, we achieved a score of {score}/{len(data)}")


            results[str(coeff)] = score
            ex_time = time.time() - time_1
            tot_time = time.time() - time_0
            print(f"Evaluation time for beta={beta}: {ex_time:.4f}s")
            if tot_time >= max_time:
                print("7 hrs up")
                break

        print("Final results:", results)
        with open(f'nq_cad_dola_{str(dola_layers_good)}_{str(dola_layers_bad)}.json', 'w') as json_file:
            json.dump(results, json_file)
            print("Successfully finished the experiment")
