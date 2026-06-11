"""Adapter for SWE-bench-style aggregate summaries (e.g. evaluation.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.adapter import (
    AdapterMetadata,
    BaseEvaluationAdapter,
    SupportedLibrary,
)
from every_eval_ever.converters.common.error import AdapterError, TransformationError
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    SourceType,
)
from every_eval_ever.helpers import get_developer, get_model_id


class SWEBenchAdapter(BaseEvaluationAdapter):
    """Converts SWE-bench harness aggregate JSON summaries to every_eval_ever format."""

    DEFAULT_BENCHMARK_NAME = 'SWE-bench'
    DEFAULT_HF_REPO = 'princeton-nlp/SWE-bench'

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='swebench-adapter',
            version='0.1.0',
            supported_library_versions=['2.*'],
            description=(
                'Converts SWE-bench aggregate evaluation.json summaries '
                'to every_eval_ever format'
            ),
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.SWE_BENCH

    def _extract_model_info(
        self,
        raw_data: Dict[str, Any],
        metadata_args: Dict[str, Any],
    ) -> ModelInfo:
        model_id_input = (metadata_args.get('model_id') or '').strip()
        if not model_id_input:
            raise TransformationError(
                'model_id is required in metadata (SWE-bench summary JSON has no model field)'
            )

        developer = get_developer(model_id_input)
        model_id = get_model_id(model_id_input, developer)

        additional: Dict[str, str] = {}
        if raw_data.get('schema_version') is not None:
            additional['swe_summary_schema_version'] = str(
                raw_data['schema_version']
            )
        fn = metadata_args.get('input_filename')
        if fn:
            additional['source_summary_file'] = str(fn)

        return ModelInfo(
            name=model_id_input,
            id=model_id,
            developer=developer if developer != 'unknown' else None,
            additional_details=additional if additional else None,
        )

    def _build_evaluation_results(
        self,
        raw_data: Dict[str, Any],
        metadata_args: Dict[str, Any],
    ) -> List[EvaluationResult]:

        total = raw_data.get("total_instances",0)
        
        resolved = raw_data.get("resolved_instances",0)
        submitted = raw_data.get("submitted_instances",0)
        completed = raw_data.get("completed_instances",0)
        unresolved = raw_data.get("unresolved_instances",0)
        empty_patch = raw_data.get("empty_patch_instances",0)
        errors = raw_data.get("error_instances",0)

        score = (resolved / total) if total else 0.0

        benchmark_name = (
            metadata_args.get('benchmark_name') or self.DEFAULT_BENCHMARK_NAME
        )
        hf_repo = metadata_args.get('hf_repo') or self.DEFAULT_HF_REPO

        details: Dict[str, str] = {
            'total_instances': str(total),
            'resolved_instances': str(resolved),
            'submitted_instances': str(submitted),
            'completed_instances': str(completed),
            'unresolved_instances': str(unresolved),
            'empty_patch_instances': str(empty_patch),
            'error_instances': str(errors),
        }

        completed_ids = raw_data.get("completed_ids", [])
        submitted_ids = raw_data.get("submitted_ids", [])
        resolved_ids = raw_data.get("resolved_ids", [])

        source_data = SourceDataHf(
            dataset_name=benchmark_name,
            source_type='hf_dataset',
            hf_repo=hf_repo,
            hf_split='test',
            samples_number=total,
        )

        return [
            EvaluationResult(
                evaluation_name=benchmark_name,
                source_data=source_data,
                metric_config=MetricConfig(
                    evaluation_description=(
                        'Resolve rate: fraction of benchmark instances where the '
                        'submitted patch passes verification (0.0–1.0)'
                    ),
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(
                    score=score,
                    details=details,
                    completed_ids=completed_ids,
                    submitted_ids=submitted_ids,
                    resolved_ids=resolved_ids,
                ),
                generation_config=GenerationConfig(
                    generation_args=GenerationArgs(
                        agentic_eval_config=AgenticEvalConfig(
                            available_tools=[AvailableTool(name='bash')],
                        ),
                    ),
                ),
            )
        ]

    def _transform_single(
        self, raw_data: Any, metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        if not isinstance(raw_data, dict):
            raise TransformationError(
                f'Expected a JSON object, got {type(raw_data).__name__}'
            )

        # Minimal validation: harness summaries always include these keys
        if 'total_instances' not in raw_data or 'resolved_instances' not in raw_data:
            raise TransformationError(
                'SWE-bench summary must include total_instances and resolved_instances'
            )

        model_info = self._extract_model_info(raw_data, metadata_args)
        evaluation_results = self._build_evaluation_results(
            raw_data, metadata_args
        )

        retrieved_timestamp = get_current_unix_timestamp()

        benchmark_name = (
            metadata_args.get('benchmark_name') or self.DEFAULT_BENCHMARK_NAME
        )

        bench_slug = benchmark_name.replace('/', '_')
        model_slug = model_info.id.replace('/', '_')
        input_slug = str(metadata_args.get('input_filename') or 'evaluation')
     
        evaluation_id = (
            f'{bench_slug}/{model_slug}/{input_slug}/{retrieved_timestamp}'
        )

        evaluator_rel_str = metadata_args.get(
            'evaluator_relationship', 'third_party'
        )
        evaluator_relationship = EvaluatorRelationship(evaluator_rel_str)

        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name', 'swebench'),
            version=metadata_args.get('eval_library_version', 'unknown'),
        )

        source_metadata = SourceMetadata(
            source_name='SWE-bench',
            source_type=SourceType.evaluation_run,
            source_organization_name=metadata_args.get(
                'source_organization_name', ''
            ),
            source_organization_url=metadata_args.get(
                'source_organization_url'
            ),
            source_organization_logo_url=metadata_args.get(
                'source_organization_logo_url'
            ),
            evaluator_relationship=evaluator_relationship,
        )

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=None,
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        file_path = Path(file_path)
        raw_data = self._load_file(file_path)
        merged_meta: Dict[str, Any] = {
            **metadata_args,
            'input_filename': file_path.name,
            'parent_eval_output_dir': str(file_path.parent),
        }
        return [self._transform_single(raw_data, merged_meta)]

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform all json results files in a directory."""
        dir_path = Path(dir_path)
        results_files = sorted(dir_path.glob('**/*.json'))

        all_logs = []
        for results_file in results_files:
            logs = self.transform_from_file(results_file, metadata_args)
            all_logs.extend(logs)

        return all_logs