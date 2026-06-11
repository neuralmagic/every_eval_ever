import shutil
import tempfile
from pathlib import Path

from every_eval_ever.converters.common.adapter import AdapterMetadata
from every_eval_ever.converters.lighteval.adapter import LightEvalAdapter
from every_eval_ever.converters.lighteval.utils import parse_model_name
from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    SourceDataHf,
)

DATA_DIR = Path('tests/data/lighteval')
RESULTS_FILE = DATA_DIR / 'results_2026-05-11T07-33-49.106520.json'


def _make_metadata_args(**overrides):
    args = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.first_party,
    }
    args.update(overrides)
    return args


def _log_for_task(logs: list[EvaluationLog], evaluation_name: str) -> EvaluationLog:
    for log in logs:
        if not log.evaluation_results:
            continue
        if log.evaluation_results[0].evaluation_name == evaluation_name:
            return log
    raise AssertionError(f'no log for task {evaluation_name!r}')


# ── Utility tests ──────────────────────────────────────────────────────


def test_parse_model_name_hosted_vllm():
    provider, path = parse_model_name(
        'hosted_vllm/inference-optimization/Qwen3-235B-A22B-Thinking-2507.w4a16'
    )
    assert provider == 'hosted_vllm'
    assert path == 'inference-optimization/Qwen3-235B-A22B-Thinking-2507.w4a16'


def test_parse_model_name_empty():
    assert parse_model_name('') == ('unknown', 'unknown')
    assert parse_model_name(None) == ('unknown', 'unknown')


def test_parse_model_name_no_slash():
    provider, path = parse_model_name('local-checkpoint')
    assert provider == 'unknown'
    assert path == 'local-checkpoint'


def test_parse_model_name_transformers():
    provider, path = parse_model_name(
        'transformers/EleutherAI/pythia-70m'
    )
    assert provider == 'transformers'
    assert path == 'EleutherAI/pythia-70m'


# ── Adapter: _get_tasks ─────────────────────────────────────────────────


def test_get_tasks_skips_all_aggregation():
    adapter = LightEvalAdapter()
    raw = {
        'results': {
            'all': {'acc': 0.5},
            'aime25|0': {'pass@k:k=1&n=1': 0.1},
            'gsm8k|0': {'exact_match': 0.9},
        }
    }
    tasks = adapter._get_tasks(raw)
    assert tasks == ['aime25|0', 'gsm8k|0']


# ── Adapter: transform_from_file ───────────────────────────────────────


def test_transform_from_file_returns_one_log_per_task():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert len(logs) == 3
    for log in logs:
        assert isinstance(log, EvaluationLog)


def test_transform_from_file_model_info():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    model = logs[0].model_info

    assert (
        model.name
        == 'inference-optimization/Qwen3-235B-A22B-Thinking-2507.w4a16'
    )
    assert model.id == model.name
    assert model.developer == 'inference-optimization'
    assert model.inference_platform is None
    assert model.inference_engine.name == 'vllm'
    assert model.additional_details['provider'] == 'hosted_vllm'
    assert model.additional_details['concurrent_requests'] == '16'


def test_transform_from_file_source_metadata():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    src = logs[0].source_metadata

    assert src.source_name == 'lighteval'
    assert src.source_type.value == 'evaluation_run'
    assert src.source_organization_name == 'TestOrg'


def test_transform_from_file_eval_library_version_unknown_sha():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert logs[0].eval_library.name == 'lighteval'
    assert logs[0].eval_library.version == 'unknown'


def test_transform_from_file_source_data_hf():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    gpqa = _log_for_task(logs, 'gpqa:diamond')
    sd = gpqa.evaluation_results[0].source_data
    assert isinstance(sd, SourceDataHf)
    assert sd.hf_repo == 'Idavidrein/gpqa'
    assert sd.hf_split == 'train'

    aime = _log_for_task(logs, 'aime25')
    sd_a = aime.evaluation_results[0].source_data
    assert isinstance(sd_a, SourceDataHf)
    assert sd_a.hf_repo == 'yentinglin/aime_2025'
    assert sd_a.hf_split == 'train'

    math = _log_for_task(logs, 'math_500')
    sd_m = math.evaluation_results[0].source_data
    assert isinstance(sd_m, SourceDataHf)
    assert sd_m.hf_repo == 'HuggingFaceH4/MATH-500'
    assert sd_m.hf_split == 'test'


def test_transform_from_file_evaluation_results_scores_and_bounds():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    gpqa = _log_for_task(logs, 'gpqa:diamond')
    assert len(gpqa.evaluation_results) == 1
    g0 = gpqa.evaluation_results[0]
    assert g0.metric_config.evaluation_description == 'gpqa_pass@k:k=1'
    assert g0.score_details.score == 0.7525252525252525
    assert g0.metric_config.lower_is_better is False
    assert g0.metric_config.min_score == 0.0
    assert g0.metric_config.max_score == 1.0

    aime = _log_for_task(logs, 'aime25')
    assert len(aime.evaluation_results) == 2
    pass_at_k = next(
        r
        for r in aime.evaluation_results
        if r.metric_config.evaluation_description.startswith('pass@k')
    )
    assert pass_at_k.score_details.score == 0.8333333333333334
    assert pass_at_k.metric_config.lower_is_better is False

    avg_n = next(
        r
        for r in aime.evaluation_results
        if r.metric_config.evaluation_description.startswith('avg@n')
    )
    assert avg_n.score_details.score == 0.8333333333333334

    math = _log_for_task(logs, 'math_500')
    assert len(math.evaluation_results) == 1
    assert math.evaluation_results[0].score_details.score == 0.904


def test_transform_from_file_uncertainty_stderr():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    aime = _log_for_task(logs, 'aime25')
    pass_at_k = next(
        r
        for r in aime.evaluation_results
        if r.metric_config.evaluation_description.startswith('pass@k')
    )
    u = pass_at_k.score_details.uncertainty
    assert u is not None
    assert u.standard_error.value == 0.06920456654478331
    assert u.standard_error.method == 'bootstrap'
    assert u.num_samples is None


def test_transform_from_file_uncertainty_stderr_without_num_samples():
    """effective_num_docs -1 yields no num_samples on Uncertainty."""
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    gpqa = _log_for_task(logs, 'gpqa:diamond')
    u = gpqa.evaluation_results[0].score_details.uncertainty
    assert u is not None
    assert u.standard_error is not None
    assert u.standard_error.value == 0.03074630074212453
    assert u.num_samples is None


def test_transform_from_file_generation_config():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    gen = logs[0].evaluation_results[0].generation_config
    assert gen is not None
    assert gen.generation_args.temperature == 0.6
    assert gen.generation_args.max_tokens == 32000
    assert gen.generation_args.top_p == 0.95
    assert gen.generation_args.top_k == 20


def test_transform_from_file_eval_timestamp():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert logs[0].evaluation_timestamp == '7455098'


# ── Adapter: transform_from_directory ────────────────────────────────────


def test_transform_from_directory():
    adapter = LightEvalAdapter()
    logs = adapter.transform_from_directory(DATA_DIR, _make_metadata_args())
    assert len(logs) == 3
    task_names = {
        r.evaluation_name for log in logs for r in log.evaluation_results
    }
    assert task_names == {'gpqa:diamond', 'aime25', 'math_500'}


# ── Adapter: inference engine override ─────────────────────────────────


def test_inference_engine_override():
    adapter = LightEvalAdapter()
    metadata = _make_metadata_args(
        inference_engine='sglang', inference_engine_version='0.4.1'
    )
    logs = adapter.transform_from_file(RESULTS_FILE, metadata)
    assert logs[0].model_info.inference_engine.name == 'sglang'
    assert logs[0].model_info.inference_engine.version == '0.4.1'


def test_eval_library_version_cli_override():
    adapter = LightEvalAdapter()
    metadata = _make_metadata_args(eval_library_version='9.9.9')
    logs = adapter.transform_from_file(RESULTS_FILE, metadata)
    assert logs[0].eval_library.version == '9.9.9'


# ── Adapter: public adapter metadata ───────────────────────────────────


def test_adapter_metadata_property():
    adapter = LightEvalAdapter()
    meta = adapter.metadata
    assert isinstance(meta, AdapterMetadata)
    assert meta.name == 'lighteval-adapter'
    assert meta.version == '0.1.0'
    assert 'lighteval' in meta.description.lower()


def test_transform_from_directory_finds_nested_results_file():
    """Recursive glob should pick up results under a nested run directory."""
    adapter = LightEvalAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / 'run_a'
        nested.mkdir()
        dest = nested / RESULTS_FILE.name
        shutil.copy(RESULTS_FILE, dest)
        logs = adapter.transform_from_directory(tmpdir, _make_metadata_args())
        assert len(logs) == 3
        task_names = {
            r.evaluation_name for log in logs for r in log.evaluation_results
        }
        assert task_names == {'gpqa:diamond', 'aime25', 'math_500'}
