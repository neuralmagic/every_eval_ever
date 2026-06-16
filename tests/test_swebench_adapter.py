import shutil
import tempfile
from pathlib import Path

import pytest

from every_eval_ever.converters.common.error import TransformationError
from every_eval_ever.converters.swebench.adapter import SWEBenchAdapter
from every_eval_ever.eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    SourceDataHf,
)

DATA_DIR = Path('tests/data/swebench')
EVALUATION_FILE = DATA_DIR / 'evaluation.json'


def _make_metadata_args(**overrides):
    args = {
        'model_id': 'Qwen/Qwen3-Coder-Next',
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.third_party,
    }
    args.update(overrides)
    return args


# ── Adapter: transform_from_file ─────────────────────────────────────────


def test_transform_from_file_returns_single_log():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    assert len(logs) == 1
    assert isinstance(logs[0], EvaluationLog)


def test_transform_from_file_model_info():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    model = logs[0].model_info

    assert model.name == 'Qwen/Qwen3-Coder-Next'
    assert model.id == 'Qwen/Qwen3-Coder-Next'
    assert model.developer == 'Qwen'
    assert model.additional_details is not None
    assert model.additional_details['swe_summary_schema_version'] == '2'
    assert model.additional_details['source_summary_file'] == 'evaluation.json'


def test_transform_from_file_source_metadata():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    src = logs[0].source_metadata

    assert src.source_name == 'SWE-bench'
    assert src.source_type.value == 'evaluation_run'
    assert src.source_organization_name == 'TestOrg'
    assert src.evaluator_relationship == EvaluatorRelationship.third_party


def test_transform_from_file_eval_library_defaults():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    assert logs[0].eval_library.name == 'swebench'
    assert logs[0].eval_library.version == 'unknown'


def test_transform_from_file_evaluation_timestamp_none():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    assert logs[0].evaluation_timestamp is None


def test_transform_from_file_evaluation_id_shape():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    parts = logs[0].evaluation_id.split('/')
    assert parts[0] == 'SWE-bench'
    assert parts[1] == 'Qwen_Qwen3-Coder-Next'
    assert parts[2] == 'evaluation.json'


def test_transform_from_file_resolve_rate_and_details():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    res = logs[0].evaluation_results[0]
    assert res.evaluation_name == 'SWE-bench'
    assert res.score_details.score == pytest.approx(156 / 300)
    d = res.score_details.details
    assert d is not None
    assert d['total_instances'] == '300'
    assert d['resolved_instances'] == '156'
    assert d['submitted_instances'] == '300'
    assert d['completed_instances'] == '252'
    assert d['unresolved_instances'] == '96'
    assert d['empty_patch_instances'] == '4'
    assert d['error_instances'] == '44'


def test_transform_from_file_instance_id_lists():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    sd = logs[0].evaluation_results[0].score_details
    assert sd.completed_ids is not None
    assert len(sd.completed_ids) == 252
    assert sd.submitted_ids is not None
    assert len(sd.submitted_ids) == 300
    assert sd.resolved_ids is not None
    assert len(sd.resolved_ids) == 156
    assert sd.completed_ids[0] == 'astropy__astropy-12907'
    assert sd.resolved_ids[0] == 'astropy__astropy-12907'


def test_transform_from_file_source_data_hf_defaults():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    src = logs[0].evaluation_results[0].source_data
    assert isinstance(src, SourceDataHf)
    assert src.dataset_name == 'SWE-bench'
    assert src.hf_repo == 'princeton-nlp/SWE-bench'
    assert src.hf_split == 'test'
    assert src.samples_number == 300


def test_transform_from_file_metric_config():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    mc = logs[0].evaluation_results[0].metric_config
    assert 'Resolve rate' in mc.evaluation_description
    assert mc.lower_is_better is False
    assert mc.min_score == 0.0
    assert mc.max_score == 1.0


def test_transform_from_file_generation_config_agentic():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_file(
        EVALUATION_FILE, _make_metadata_args()
    )
    gen = logs[0].evaluation_results[0].generation_config
    assert gen is not None
    assert gen.generation_args.agentic_eval_config is not None
    tools = gen.generation_args.agentic_eval_config.available_tools
    assert tools is not None
    assert len(tools) == 1
    assert tools[0].name == 'bash'


# ── Adapter: transform_from_directory ────────────────────────────────────


def test_transform_from_directory():
    adapter = SWEBenchAdapter()
    logs = adapter.transform_from_directory(
        DATA_DIR, _make_metadata_args()
    )
    assert len(logs) == 1


def test_transform_from_directory_nested_file():
    adapter = SWEBenchAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / 'run_a'
        nested.mkdir()
        dest = nested / EVALUATION_FILE.name
        shutil.copy(EVALUATION_FILE, dest)
        logs = adapter.transform_from_directory(
            tmpdir, _make_metadata_args()
        )
        assert len(logs) == 1
        assert (
            logs[0].model_info.additional_details['source_summary_file']
            == 'evaluation.json'
        )


# ── Adapter: metadata overrides ──────────────────────────────────────────


def test_benchmark_name_and_hf_repo_override():
    adapter = SWEBenchAdapter()
    meta = _make_metadata_args(
        benchmark_name='SWE-bench-Lite',
        hf_repo='princeton-nlp/SWE-bench_Lite',
    )
    logs = adapter.transform_from_file(EVALUATION_FILE, meta)
    res = logs[0].evaluation_results[0]
    assert res.evaluation_name == 'SWE-bench-Lite'
    assert isinstance(res.source_data, SourceDataHf)
    assert res.source_data.dataset_name == 'SWE-bench-Lite'
    assert res.source_data.hf_repo == 'princeton-nlp/SWE-bench_Lite'


def test_eval_library_version_override():
    adapter = SWEBenchAdapter()
    meta = _make_metadata_args(eval_library_version='2.0.0')
    logs = adapter.transform_from_file(EVALUATION_FILE, meta)
    assert logs[0].eval_library.version == '2.0.0'


# ── Adapter: validation errors ───────────────────────────────────────────


def test_transform_requires_model_id():
    adapter = SWEBenchAdapter()
    bad_meta = {
        'source_organization_name': 'TestOrg',
        'evaluator_relationship': EvaluatorRelationship.third_party,
    }
    with pytest.raises(TransformationError, match='model_id is required'):
        adapter.transform_from_file(EVALUATION_FILE, bad_meta)


def test_transform_requires_total_and_resolved():
    adapter = SWEBenchAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / 'bad.json'
        p.write_text('{"resolved_instances": 1}', encoding='utf-8')
        with pytest.raises(TransformationError, match='total_instances'):
            adapter.transform_from_file(p, _make_metadata_args())


def test_transform_single_rejects_non_object():
    adapter = SWEBenchAdapter()
    with pytest.raises(TransformationError, match='JSON object'):
        adapter._transform_single([], _make_metadata_args())
