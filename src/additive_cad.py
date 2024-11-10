"""
The additive_cad module: includes the AdditiveCad class which allows simple llm inference with
original CAD as well as other decoding strategies.
"""

from typing import Literal, Any, Tuple, Union, List, Dict
import sys
import time
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.generation.utils import ModelOutput
from src.experiment_types import ExperimentConfig
from src.utils import normalize_answer, evaluate_recall, evaluate_em
from dataclasses import fields


class AdditiveCad:
    """
    Sets up an LLM that we can do inference on.
    """
    def __init__(self, config: ExperimentConfig):
        self.config: ExperimentConfig = config
        self.device: Literal['cpu', 'cuda'] = config.device
        self.tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(config.llm_name)
        self.model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.llm_name,
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
        apc: float = 0.1,
    ) -> int:
        """
        Take 2 distributions, do contrastive decoding with adaptive plausibility constraint
        then return the token id with highest logit. Beta defaults to the literature value.
        """
        good_probs = torch.softmax(good_distribution, dim=-1)
        thresh = apc * float(torch.max(good_probs).item())
        implausible_ids = (good_probs < thresh).nonzero(as_tuple=False)

        # Masking for the APC
        good_distribution[implausible_ids[:, 0], implausible_ids[:, 1]] = -1000  # Essentially -inf
        bad_distribution = torch.where(
            bad_distribution == float('-inf'),
            torch.tensor(-1000.0),
            bad_distribution
        )

        # CAD
        logits = (1 + beta) * good_distribution - beta * bad_distribution
        return int(torch.argmax(logits).item())

    def add_cad_decoding(
        self,
        bad_distribution: torch.Tensor,
        good_distribution: torch.Tensor,
        gamma: float = 0.0,
    ) -> int:
        """
        Take 2 distributions, do additive contrastive decoding with adaptive plausibility constraint
        then return the token id with highest logit. Gamma defaults to 0.0.
        """
        good_probs = torch.softmax(good_distribution, dim=-1)
        bad_probs = torch.softmax(bad_distribution, dim=-1)

        # Additve CAD equation
        new_probs = bad_probs + (good_probs - bad_probs) * (10**gamma)
        return int(torch.argmax(new_probs).item())

    def generate_response(
        self,
        context: str,
        question: str,
        coeff: float,
        apc: float = 0.1,
    ) -> Union[str, None]:
        """
        Return the generated response to a question given a question and context, based on the
        settings of the `AdditiveCad` object config.

        Args:
            context (str):
                The context to be placed in the `{context}` formatter of the prompts of the config
                file.
            question (str):
                The question to be placed in the `{question}` formatter of the prompts of the config
                file.
            coeff (float):
                The coefficient required for the configured contrastive decoding type.
            apc (float):
                The adaptive plausibility constraint value, defaults to 0.1. Only relevant for
                regular CAD.

        Returns:
            The LLM's response to a question. Returns `str` type if successful, or `None` if an
            error occured.
        """
        output: str = " "

        for token_number in range(self.config.max_tokens):
            good_distribution = self.generate_distribution(
                input_text=self.config.context_prompt.format(
                    context=context,
                    question=question
                ) + output,
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
                        apc=apc,
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
                if next_token_id == 2:
                    # EOS has been reached
                    return output

                output = self.tokenizer.decode(
                    self.tokenizer.encode(output) + [next_token_id],
                    skip_special_tokens=False
                )

                # stopping_symbols = [".", "\n"]
                # for stopping_symbol in stopping_symbols:
                #     if stopping_symbol in output:
                #         return output
            else:
                return None
        return output  # Assuming the space was taken out before context and prompt passed in

    def log_config(self):
        "Print the config contents to stdout and flush"
        print("\n\n----------Begin Experiment----------")
        for field in fields(self.config):
            value = getattr(self.config, field.name)
            print(f"{field.name} = {value}")
        sys.stdout.flush()

    def generate_results(
        self
    ) -> Dict[Literal['EM', 'Recall'], Dict[str, int]]:
        """
        Generate a set of results for an entire experiment based on the config of the object.

        Returns:
            A dictionary of scores for each coefficient, for EM and Recall metrics.
            Example: `{'EM': {'0.0': 167, '1.0': 193}, 'Recall': {'0.0': 267, '1.0': 293}}`
        """

        self.log_config()

        with open(self.config.dataset_path, 'r') as file:
            data: List[Any] = json.load(file)

        time_0 = time.time()
        em_results: Dict[str, int] = {}
        recall_results: Dict[str, int] = {}

        # Loop through every test coefficient for the experiment
        for coeff in self.config.test_coefficients:
            time_1 = time.time()
            em_score: int = 0
            recall_score: int = 0
            # Loop through each question
            for idx, qa in enumerate(data):
                context: str = qa["context"]
                question: str = qa["question"]
                answers: List[str] = qa["answer"]  # There may be multiple answers

                response = self.generate_response(
                    context=context,
                    question=question,
                    coeff=coeff,
                    apc=self.config.apc
                )

                if response is None:  # Error with CAD generation
                    return {}
                response = normalize_answer(response)

                # For debugging, print an example output every 100 questions
                if idx % 100 == 0:
                    print(f"{idx}. CAD answer: {repr(response)}")
                    print(
                        f"{idx}. Correct answers:",
                        " ".join([repr(normalize_answer(answers[i])) for i in range(len(answers))])
                    )
                    sys.stdout.flush()
                if evaluate_em(response, answers):
                    em_score += 1
                if evaluate_recall(response, answers):
                    recall_score += 1

                # In danger of time elapsing
                tot_time = (time.time() - time_0) / 3600
                if tot_time >= self.config.max_hours - 0.1:
                    print("----------")
                    print(f"Out of time on question {idx} with coeff {coeff}")
                    print(f"EM and Recall scores: {em_score} and {recall_score}")
                    print(f"Total eval time {tot_time:.2f} hrs")
                    print("Final EM results:", em_results)
                    print("Final Recall results:", recall_results)
                    print("----------")
                    sys.stdout.flush()
                    return {
                        'EM': em_results,
                        'Recall': recall_results,
                    }

            em_results[str(coeff)] = em_score
            recall_results[str(coeff)] = recall_score
            ex_time = (time.time() - time_1) / 3600

            print("----------")
            print(f"Result for coeff {coeff} (eval time {ex_time:.2f} hrs)")
            print(f"EM score       {em_score} /{len(data)}")
            print(f"Recall score   {recall_score} /{len(data)}")
            print("----------")
            sys.stdout.flush()  # Ensure we log to the .out file

        print("Final EM results:", em_results)
        print("Final Recall results:", recall_results)
        sys.stdout.flush()
        return {
            'EM': em_results,
            'Recall': recall_results,
        }
