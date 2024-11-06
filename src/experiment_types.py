"""
The experiment_types module: defines the JSON schema of the experiment config file.
"""

from typing import Literal, List
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """
    Settings of the experiment.

    Attributes:
        llm_name (str):
            The name of the language model. Should be a valid huggingface model name, e.g., "huggyllama/llama-7b".
        device (Literal['cpu', 'cuda']):
            The computation device to use. Must be either 'cpu' or 'cuda'.
        context_prompt (str):
            The template prompt string used to create prompts with context. Requires {context} and {question} formatters.
            Example: `"{context} Using only the references listed above, answer the following question: Question: {question}. Answer:"`.
        no_context_prompt (str):
            The template prompt string used when no context is provided. Requires {question} formatter.
            Example: `"Answer the following question: Question: {question}. Answer:"`
        decoding_strategy (Literal['CAD', 'additive-CAD']):
            The decoding strategy to use. Must be either 'CAD' or 'additive-CAD'.
        test_coefficients (List[float]):
            The primary coefficients to test. These should match the hyperparameter of the selected decoding strategy.
        apc (float):
            The adaptive plausibility constraint (APC) to use. This rejects all tokens less than this proportion of the probability of the maximum token of the distribution with context. Set to `0` for no APC.
        dola_layers_context (Literal['high', 'low', 'none']):
            The DoLa layers setting for the distribution with context. Set to `'none'` for no DoLa on this distribution.
        dola_layers_no_context (Literal['high', 'low', 'none']):
            The DoLa layers setting for the distribution with no context. Set to `'none'` for no DoLa on this distribution.
        max_tokens (int):
            After how many tokens to stop generating any more output (to save on compute).
        dataset_path (str):
            The file path location of the JSON question-answer dataset.
    """
    llm_name: str
    device: Literal['cpu', 'cuda']
    context_prompt: str
    no_context_prompt: str
    decoding_strategy: Literal['CAD', 'additive-CAD']
    test_coefficients: List[float]
    apc: float
    dola_layers_context: Literal['high', 'low', 'none']
    dola_layers_no_context: Literal['high', 'low', 'none']
    max_tokens: int
    dataset_path: str

    @classmethod
    def from_dict(cls, config_dict):
        """Class method to create ExperimentConfig instance from a dictionary."""
        return cls(**config_dict)
