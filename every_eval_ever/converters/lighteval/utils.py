"""Utility functions for the lighteval adapter."""

from pathlib import Path
from typing import Dict, Optional


def parse_model_name(model_name: str) -> tuple[str, str]:
    """Parse lighteval model_name into provider and model path.

    lighteval uses format like: "hosted_vllm/mistralai/Ministral-3-14B-Instruct-2512"
    Returns: (provider, model_path)
    """
    if not model_name:
        return ('unknown', 'unknown')

    parts = model_name.split('/', 1)
    if len(parts) == 2:
        provider = parts[0]
        model_path = parts[1]
        return (provider, model_path)

    return ('unknown', model_name)


# Maps lighteval provider values to inference platform
PROVIDER_TO_INFERENCE_PLATFORM = {
    'openai': 'openai',
    'anthropic': 'anthropic',
    'together': 'together',
}

# Maps lighteval provider values to inference engine names
PROVIDER_TO_INFERENCE_ENGINE = {
    'vllm': 'vllm',
    'hosted_vllm': 'vllm',
    'tgi': 'text-generation-inference',
    'transformers': 'transformers',
}

# Known metric bounds: metric_name -> (min_score, max_score)
# max_score of None means unbounded
KNOWN_METRIC_BOUNDS = {
    'acc': (0.0, 1.0),
    'acc_norm': (0.0, 1.0),
    'exact_match': (0.0, 1.0),
    'f1': (0.0, 1.0),
    'em': (0.0, 1.0),
    'pass@k': (0.0, 1.0),
    'avg@n': (0.0, 1.0),
    'mcc': (-1.0, 1.0),
    'bleu': (0.0, 100.0),
    'rouge1': (0.0, 1.0),
    'rouge2': (0.0, 1.0),
    'rougeL': (0.0, 1.0),
    'rougeLsum': (0.0, 1.0),
    'ter': (0.0, None),
    'brier_score': (0.0, 1.0),
}
