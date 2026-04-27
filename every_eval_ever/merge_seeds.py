"""Utilities for merging multiple seed runs into averaged evaluation logs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluationResult,
    ScoreDetails,
    StandardError,
    Uncertainty,
)


def _extract_seed_from_log(log: EvaluationLog) -> Optional[int]:
    """Extract seed value from model_info.additional_details or generation_config."""
    # Try generation_config.generation_args.seed first (lighteval stores it here)
    if log.evaluation_results:
        for result in log.evaluation_results:
            if result.generation_config:
                # Check generation_args
                if result.generation_config.generation_args:
                    gen_args = result.generation_config.generation_args
                    if hasattr(gen_args, 'seed') and gen_args.seed is not None:
                        try:
                            return int(gen_args.seed)
                        except (ValueError, TypeError):
                            pass

                # Check additional_details
                if result.generation_config.additional_details:
                    gen_details = result.generation_config.additional_details
                    for key in ['seed', 'random_seed']:
                        if key in gen_details:
                            try:
                                seed_str = gen_details[key]
                                # Handle JSON-encoded values
                                if isinstance(seed_str, str):
                                    if seed_str.isdigit():
                                        return int(seed_str)
                                    # Try parsing JSON
                                    try:
                                        seed_val = json.loads(seed_str)
                                        return int(seed_val)
                                    except (json.JSONDecodeError, ValueError, TypeError):
                                        pass
                                return int(seed_str)
                            except (ValueError, TypeError):
                                pass

    # Try model_info additional_details
    if log.model_info.additional_details:
        for key in ['seed', 'random_seed']:
            if key in log.model_info.additional_details:
                try:
                    return int(log.model_info.additional_details[key])
                except (ValueError, TypeError):
                    pass

    return None


def _make_grouping_key(log: EvaluationLog, result: EvaluationResult) -> tuple:
    """Create a grouping key for matching results across seeds."""
    return (
        log.model_info.id,
        result.evaluation_name,
        result.metric_config.evaluation_description,
        result.source_data.dataset_name if result.source_data else None,
    )


def merge_seed_runs(
    logs: List[EvaluationLog],
    num_input_paths: Optional[int] = None,
) -> List[EvaluationLog]:
    """Merge multiple seed runs into averaged evaluation logs.

    Args:
        logs: List of EvaluationLog objects from different seed runs (often one
            log per benchmark *task* per results file).
        num_input_paths: Number of distinct input result files (seeds). When each
            file defines many tasks, this must be ``len(paths)``, not ``len(logs)``.
            If omitted, defaults to ``len(logs)`` (only correct when each input
            file yields exactly one EvaluationLog).

    Returns:
        List containing a single merged EvaluationLog with averaged scores

    Raises:
        ValueError: If logs have incompatible configurations
    """
    if not logs:
        return []

    if len(logs) == 1:
        return logs

    expected_group_size = (
        num_input_paths if num_input_paths is not None else len(logs)
    )

    # Validate that all logs are for the same model
    model_ids = {log.model_info.id for log in logs}
    if len(model_ids) > 1:
        raise ValueError(
            f"Cannot merge logs from different models: {model_ids}"
        )

    # Group evaluation results across all logs by (model, task, metric)
    grouped_results: Dict[tuple, List[tuple[EvaluationResult, int, EvaluationLog]]] = defaultdict(list)

    for log in logs:
        seed = _extract_seed_from_log(log)
        for result in log.evaluation_results:
            key = _make_grouping_key(log, result)
            grouped_results[key].append((result, seed, log))

    # Compute averaged results
    merged_results = []

    for key, result_group in grouped_results.items():
        if len(result_group) != expected_group_size:
            # Not all seeds have this metric - skip or warn
            continue

        # Extract scores and seeds
        scores = [r.score_details.score for r, _, _ in result_group]
        seed_values = [s for _, s, _ in result_group if s is not None]

        # Compute statistics
        mean_score = statistics.mean(scores)
        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            std_error = std_dev / (len(scores) ** 0.5)
        else:
            std_error = 0.0

        # Use the first result as template
        template_result, _, _ = result_group[0]

        # Build uncertainty with seed-based standard error
        uncertainty = Uncertainty(
            standard_error=StandardError(
                value=std_error,
                method='across_seeds',
            ),
            num_samples=len(scores),
        )

        # Create merged result with additional seed details
        additional_seed_details = {
            'seed_scores': json.dumps(scores),
            'seed_values': json.dumps(seed_values) if seed_values else json.dumps([]),
        }

        # Preserve original details if any, and add seed info
        score_details_dict = template_result.score_details.details or {}
        score_details_dict.update(additional_seed_details)

        merged_result = EvaluationResult(
            evaluation_name=template_result.evaluation_name,
            source_data=template_result.source_data,
            evaluation_timestamp=template_result.evaluation_timestamp,
            metric_config=template_result.metric_config,
            score_details=ScoreDetails(
                score=mean_score,
                uncertainty=uncertainty,
                details=score_details_dict,
            ),
            generation_config=template_result.generation_config,
        )
        merged_results.append(merged_result)

    # Create merged log using first log as template
    template_log = logs[0]

    # Compute time range
    start_times = []
    end_times = []

    for log in logs:
        if log.evaluation_timestamp:
            try:
                start_times.append(int(log.evaluation_timestamp))
            except (ValueError, TypeError):
                pass
        # Try to extract end time from additional_details if available
        if (log.model_info.additional_details and
            'end_time' in log.model_info.additional_details):
            try:
                end_times.append(int(log.model_info.additional_details['end_time']))
            except (ValueError, TypeError):
                pass

    # Use earliest start time as evaluation_timestamp
    eval_timestamp = str(min(start_times)) if start_times else template_log.evaluation_timestamp

    # Add time range to model_info additional_details
    model_additional = dict(template_log.model_info.additional_details or {})
    if start_times and end_times:
        model_additional['evaluation_time_range'] = json.dumps({
            'start_time': min(start_times),
            'end_time': max(end_times),
        })
    model_additional['num_seeds_merged'] = str(expected_group_size)

    # Create merged model_info
    merged_model_info = template_log.model_info.model_copy(
        update={'additional_details': model_additional}
    )

    # Create merged evaluation log
    merged_log = EvaluationLog(
        schema_version=template_log.schema_version,
        evaluation_id=template_log.evaluation_id,
        retrieved_timestamp=template_log.retrieved_timestamp,
        evaluation_timestamp=eval_timestamp,
        source_metadata=template_log.source_metadata,
        eval_library=template_log.eval_library,
        model_info=merged_model_info,
        evaluation_results=merged_results,
        detailed_evaluation_results=None,  # Don't merge instance-level results
    )

    return [merged_log]
