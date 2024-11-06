"""
The utils module: Defines some helper functions including the definition of exact match (EM)
"""

from typing import List
import re
import string


def normalize_answer(s: str) -> str:
    "Lower text and remove punctuation, articles and extra whitespace."
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


def evaluate_em(
        prediction: str,
        answers: List[str],
    ) -> bool:
    "Exact match (EM) metric given a list of correct answers"
    norm_prediction = normalize_answer(prediction)
    norm_answers = [normalize_answer(ans) for ans in answers]

    for norm_answer in norm_answers:
        if norm_answer == norm_prediction:
            return True
    return False


def evaluate_recall(
        prediction: str,
        answers: List[str],
    ) -> bool:
    "Recall metric (not exact match) given a list of correct answers"
    norm_prediction = normalize_answer(prediction)
    norm_answers = [normalize_answer(ans) for ans in answers]

    for norm_answer in norm_answers:
        if norm_answer in norm_prediction:
            return True
    return False
