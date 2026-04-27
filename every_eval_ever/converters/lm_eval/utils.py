"""Utility functions for the lm-eval adapter."""

from pathlib import Path
from typing import Any, Dict, Optional


def config_run_seed_string(config: Dict[str, Any]) -> Optional[str]:
    """Return the run seed from lm-eval top-level ``config`` as a decimal string.

    Seeds live on ``config.gen_kwargs.seed`` (generation) and/or ``random_seed``
    etc., not on per-task ``configs[task].generation_kwargs``. Merge needs this
    on ``model_info.additional_details`` so :func:`merge_seed_runs` can populate
    ``seed_values`` in merged ``score_details.details``.
    """
    gen_kwargs = config.get('gen_kwargs')
    if isinstance(gen_kwargs, dict) and gen_kwargs.get('seed') is not None:
        try:
            return str(int(gen_kwargs['seed']))
        except (ValueError, TypeError):
            pass
    for key in ('random_seed', 'numpy_seed', 'torch_seed', 'fewshot_seed'):
        val = config.get(key)
        if val is not None:
            try:
                return str(int(val))
            except (ValueError, TypeError):
                pass
    return None


def normalize_lm_eval_benchmark_dataset_name(task_or_dataset: str) -> str:
    """Map per-subject lm-eval task keys to the benchmark name used for outputs.

    lm-eval registers tasks like ``mmlu_pro_chat_biology``; the CLI writes files
    under ``dataset_name`` from ``source_data``. Using the subject-specific key
    would fragment outputs; collapse to ``mmlu_pro_chat`` for all matching tasks.
    """
    if task_or_dataset.startswith('mmlu_pro_chat_'):
        return 'mmlu_pro_chat'
    return task_or_dataset


def parse_model_args(model_args: str | None) -> Dict[str, str]:
    """Parse lm-eval model_args string (comma-separated key=value pairs).

    Handles the common format: "pretrained=EleutherAI/pythia-160m,dtype=float16"
    """
    if not model_args or not isinstance(model_args, str):
        return {}
    result = {}
    for part in model_args.split(','):
        if '=' in part:
            key, value = part.split('=', 1)
            result[key.strip()] = value.strip()
        elif result:
            # Continuation of previous value that contained a comma
            last_key = list(result.keys())[-1]
            result[last_key] += ',' + part
    return result


def find_samples_file(output_dir: Path, task_name: str) -> Optional[Path]:
    """Find the samples JSONL file for a given task in the output directory.

    lm-eval writes samples as: samples_<task_name>_<datetime>.jsonl
    """
    pattern = f'samples_{task_name}_*.jsonl'
    matches = sorted(output_dir.glob(pattern))
    if matches:
        return matches[-1]  # Most recent
    # Also check subdirectories (lm-eval nests under model_name_sanitized/)
    matches = sorted(output_dir.glob(f'**/{pattern}'))
    if matches:
        return matches[-1]
    return None


# Maps lm-eval config.model values to every_eval_ever inference_platform
MODEL_TYPE_TO_INFERENCE_PLATFORM = {
    'openai-completions': 'openai',
    'openai-chat-completions': 'openai',
    'anthropic': 'anthropic',
    'anthropic-chat': 'anthropic',
    'together': 'together',
}

# Maps lm-eval config.model values to inference engine names
MODEL_TYPE_TO_INFERENCE_ENGINE = {
    'hf': 'transformers',
    'vllm': 'vllm',
    'gguf': 'llama.cpp',
}

# Known metric bounds: metric_name -> (min_score, max_score)
# max_score of None means unbounded
KNOWN_METRIC_BOUNDS = {
    'acc': (0.0, 1.0),
    'acc_norm': (0.0, 1.0),
    'exact_match': (0.0, 1.0),
    'f1': (0.0, 1.0),
    'em': (0.0, 1.0),
    'mc1': (0.0, 1.0),
    'mc2': (0.0, 1.0),
    'mcc': (-1.0, 1.0),
    'bleu': (0.0, 100.0),
    'rouge1': (0.0, 1.0),
    'rouge2': (0.0, 1.0),
    'rougeL': (0.0, 1.0),
    'rougeLsum': (0.0, 1.0),
    'ter': (0.0, None),
    'brier_score': (0.0, 1.0),
}
