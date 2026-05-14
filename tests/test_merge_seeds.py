"""Tests for every_eval_ever.merge_seeds."""

import json
import statistics

import pytest

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataPrivate,
    SourceMetadata,
)
from every_eval_ever.merge_seeds import (
    _extract_seed_from_log,
    _make_grouping_key,
    merge_seed_runs,
)


def _source_data(name: str = 'bench_ds') -> SourceDataPrivate:
    return SourceDataPrivate(dataset_name=name, source_type='other')


def _metric(description: str = 'exact_match') -> MetricConfig:
    return MetricConfig(
        evaluation_description=description,
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=0.0,
        max_score=1.0,
    )


def _result(
    score: float,
    *,
    eval_name: str = 'gsm8k',
    metric_desc: str = 'exact_match',
    dataset_name: str = 'bench_ds',
    evaluation_timestamp: str = '1700000000',
    seed_key: str = 'seed',
    seed_value: str = '42',
    score_details_details: dict | None = None,
) -> EvaluationResult:
    gen_details = {seed_key: seed_value}
    return EvaluationResult(
        evaluation_name=eval_name,
        source_data=_source_data(dataset_name),
        evaluation_timestamp=evaluation_timestamp,
        metric_config=_metric(metric_desc),
        score_details=ScoreDetails(
            score=score,
            details=score_details_details,
        ),
        generation_config=GenerationConfig(
            generation_args=None,
            additional_details=gen_details,
        ),
    )


def _log(
    *results: EvaluationResult,
    model_id: str = 'RedHatAI/Qwen3-4B-Thinking-2507',
    evaluation_id: str = 'run/1',
    retrieved_timestamp: str = '1700000001',
    evaluation_timestamp: str | None = '1700000100',
    model_end_time: str | None = None,
) -> EvaluationLog:
    add = {'end_time': model_end_time} if model_end_time else None
    return EvaluationLog(
        schema_version='0.2.0',
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=evaluation_timestamp,
        source_metadata=SourceMetadata(
            source_name='test',
            source_type='evaluation_run',
            source_organization_name='Org',
            evaluator_relationship=EvaluatorRelationship.first_party,
        ),
        eval_library=EvalLibrary(name='lm_eval', version='0.4'),
        model_info=ModelInfo(
            name=model_id,
            id=model_id,
            additional_details=add,
        ),
        evaluation_results=list(results),
    )


def test_merge_empty_returns_empty():
    assert merge_seed_runs([]) == []


def test_merge_single_log_returns_same_list():
    r = _result(0.7, seed_value='1')
    log = _log(r)
    logs_in = [log]
    out = merge_seed_runs(logs_in)
    assert out is logs_in
    assert out[0] is log


def test_extract_seed_digit_string():
    log = _log(_result(0.0, seed_value='3344'))
    assert _extract_seed_from_log(log) == 3344


def test_extract_seed_random_seed_key():
    res = _result(
        0.0,
        seed_key='random_seed',
        seed_value='99',
    )
    assert _extract_seed_from_log(_log(res)) == 99


def test_extract_seed_json_encoded_string():
    res_digit = _result(0.0, seed_value='3344')
    assert _extract_seed_from_log(_log(res_digit)) == 3344

    res_json = _result(0.0, seed_value='"3344"')
    assert _extract_seed_from_log(_log(res_json)) == 3344


def test_extract_seed_returns_none_without_generation_config():
    res = EvaluationResult(
        evaluation_name='t',
        source_data=_source_data(),
        metric_config=_metric(),
        score_details=ScoreDetails(score=0.5),
        generation_config=None,
    )
    assert _extract_seed_from_log(_log(res)) is None


def test_make_grouping_key():
    log = _log(_result(0.1, eval_name='a', metric_desc='m', dataset_name='d1'))
    key = _make_grouping_key(log, log.evaluation_results[0])
    assert key == (log.model_info.id, 'a', 'm', 'd1')


def test_merge_two_seeds_averages_score_and_stderr():
    r1 = _result(
        0.4,
        evaluation_timestamp='1700000000',
        seed_value='1',
    )
    r2 = _result(
        0.6,
        evaluation_timestamp='1700000002',
        seed_value='2',
    )
    merged = merge_seed_runs([_log(r1), _log(r2)], num_input_paths=2)
    assert len(merged) == 1
    mlog = merged[0]
    assert len(mlog.evaluation_results) == 1
    mr = mlog.evaluation_results[0]
    assert mr.score_details.score == pytest.approx(0.5)
    std_dev = statistics.stdev([0.4, 0.6])
    expected_se = std_dev / (2**0.5)
    assert mr.score_details.uncertainty.standard_error.value == pytest.approx(
        expected_se
    )
    assert mr.score_details.uncertainty.standard_error.method == 'across_seeds'
    assert mr.score_details.uncertainty.num_samples == 2

    d = mr.score_details.details
    assert json.loads(d['seed_scores']) == [0.4, 0.6]
    assert json.loads(d['seed_values']) == [1, 2]
    assert json.loads(d['evaluation_timestamps']) == [1700000000, 1700000002]


def test_merge_identical_scores_zero_standard_error():
    r1 = _result(0.5, evaluation_timestamp='1', seed_value='1')
    r2 = _result(0.5, evaluation_timestamp='2', seed_value='2')
    merged = merge_seed_runs([_log(r1), _log(r2)])
    se = merged[0].evaluation_results[0].score_details.uncertainty.standard_error
    assert se.value == 0.0


def test_merge_different_models_raises():
    r1 = _result(0.1)
    r2 = _result(0.2)
    with pytest.raises(ValueError, match='different models'):
        merge_seed_runs(
            [
                _log(r1, model_id='a/m1'),
                _log(r2, model_id='a/m2'),
            ]
        )


def test_merge_no_complete_groups_raises():
    """Each group appears once but two inputs expected → nothing merged."""
    r1 = _result(0.1, eval_name='only_a')
    r2 = _result(0.2, eval_name='only_b')
    with pytest.raises(ValueError, match='No evaluation results could be averaged'):
        merge_seed_runs([_log(r1), _log(r2)], num_input_paths=2)


def test_merge_expected_size_larger_than_group_raises():
    r1 = _result(0.4, evaluation_timestamp='1')
    r2 = _result(0.6, evaluation_timestamp='2')
    with pytest.raises(ValueError, match='No evaluation results could be averaged'):
        merge_seed_runs([_log(r1), _log(r2)], num_input_paths=3)


def test_merge_model_info_num_seeds_and_time_range():
    r1 = _result(0.0, evaluation_timestamp='100', seed_value='1')
    r2 = _result(0.0, evaluation_timestamp='200', seed_value='2')
    merged = merge_seed_runs(
        [
            _log(r1, model_end_time='500', evaluation_timestamp='100'),
            _log(r2, model_end_time='600', evaluation_timestamp='200'),
        ],
        num_input_paths=2,
    )
    mi = merged[0].model_info
    assert mi.additional_details['num_seeds_merged'] == '2'
    assert merged[0].evaluation_timestamp == '100'
    time_range = json.loads(mi.additional_details['evaluation_time_range'])
    assert time_range == {'start_time': 100, 'end_time': 600}


def test_merge_preserves_prior_score_details_strings():
    r1 = _result(
        0.1,
        evaluation_timestamp='1',
        score_details_details={'keep': 'yes'},
    )
    r2 = _result(0.3, evaluation_timestamp='2')
    merged = merge_seed_runs([_log(r1), _log(r2)])
    d = merged[0].evaluation_results[0].score_details.details
    assert d['keep'] == 'yes'
    assert 'seed_scores' in d


def test_merge_detailed_evaluation_results_cleared():
    r1 = _result(0.1, evaluation_timestamp='1')
    r2 = _result(0.3, evaluation_timestamp='2')
    log1 = _log(r1)
    # EvaluationLog may not have detailed_evaluation_results in constructor default None
    merged = merge_seed_runs([log1, _log(r2)])
    assert merged[0].detailed_evaluation_results is None
