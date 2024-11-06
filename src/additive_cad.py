from typing import Literal, Any, Tuple, Union
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.generation.utils import ModelOutput


class AdditiveCad:
    """
    Sets up an LLM that we can do inference on.
    """
    def __init__(self, model_name: str, device: Literal['cpu', 'cuda']):
        self.model_name: str = model_name
        self.device: Literal['cpu', 'cuda'] = device
        self.tokenizer: PreTrainedTokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model: PreTrainedModel = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )

    def generate_token(
            self,
            input_text: str,
            dola_layers: Union[Literal['high', 'low'], None] = None,
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
            dola_layers=dola_layers,
            max_new_tokens=1,
            min_new_tokens=1,
        )
        if not isinstance(outputs, ModelOutput):
            return None
        if isinstance(outputs.scores, Tuple):
            if isinstance(outputs.scores[0], torch.Tensor):
                return outputs.scores[0]
        return None

    def contrastive_decoding(
            self,
            bad_distribution: torch.Tensor,
            good_distribution: torch.Tensor,
            alpha: float = 0.1,
            beta: float = 1.0,
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
        thresh = alpha * float(torch.max(good_probs).item())
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
    
    def contrastive_decoding_novel(
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

    def cad_generate_memotrap(
            self,
            context: str,
            prompt: str,
            dola_layers_good: Union[Literal['high', 'low'], None] = None,
            dola_layers_bad: Union[Literal['high', 'low'], None] = None,
            alpha: float = 0.1,
            beta: float = 1.0,
            gamma: Union[float, None] = None,
            max_tokens: int = 20,
        ) -> Union[str, None]:
        """
        Given an input context and prompt, return the CAD-generated response
        """

        for _ in range(max_tokens):
            good_dis = self.generate_token(
                input_text=context + ": " + prompt,
                dola_layers=dola_layers_good
            )
            bad_dis = self.generate_token(
                input_text=prompt,
                dola_layers=dola_layers_bad
            )
            if good_dis is not None and bad_dis is not None:
                if gamma is None:
                    next_token_id = self.contrastive_decoding(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        alpha=alpha,
                        beta=beta,
                    )
                elif gamma is not None:
                    next_token_id = self.contrastive_decoding_novel(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        gamma=gamma
                    )
                if next_token_id == -1:
                    raise TypeError("contrastive_decoding failed to return correct id")
                prompt = self.tokenizer.decode(self.tokenizer.encode(prompt) + [next_token_id], skip_special_tokens=True)

                if self.tokenizer.decode(next_token_id) == ".":
                    break  # Stop generating after the sentence is ended
            else: raise  
        return context + ": " + prompt  # Assuming the space was taken out before context and prompt passed in

    def cad_generate_nq(
            self,
            context: str,
            question: str,
            dola_layers_good: Union[Literal['high', 'low'], None] = None,
            dola_layers_bad: Union[Literal['high', 'low'], None] = None,
            alpha: float = 0.1,
            beta: float = 1.0,
            gamma: Union[float, None] = None,
            max_tokens: int = 20,
        ) -> Union[str, None]:
        """
        Given an input context and prompt, return the CAD-generated response
        """
        sys_prompt_context: str = "Instruction: read the given information and answer the corresponding question.\n\n"
        sys_prompt_no_context: str = "Instruction: answer the corresponding question.\n\n"
        output: str = " "

        for _ in range(max_tokens):
            good_dis = self.generate_token(
                input_text=sys_prompt_context + context + "\nQ: " + question + "\nA:" + output,
                dola_layers=dola_layers_good
            )
            bad_dis = self.generate_token(
                input_text=sys_prompt_no_context + "Q: " + question + "\nA:" + output,
                dola_layers=dola_layers_bad
            )
            if good_dis is not None and bad_dis is not None:
                if gamma is None:
                    next_token_id = self.contrastive_decoding(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        alpha=alpha,
                        beta=beta,
                    )
                elif gamma is not None:
                    next_token_id = self.contrastive_decoding_novel(
                        bad_distribution=bad_dis,
                        good_distribution=good_dis,
                        gamma=gamma
                    )
                if next_token_id == -1:
                    raise TypeError("contrastive_decoding failed to return correct id")
                output = self.tokenizer.decode(self.tokenizer.encode(output) + [next_token_id], skip_special_tokens=True)

                # stopping_symbols = [".", "\n"]
                # for stopping_symbol in stopping_symbols:
                #     if stopping_symbol in output:
                #         return output
            else:
                return None
        return output  # Assuming the space was taken out before context and prompt passed in
