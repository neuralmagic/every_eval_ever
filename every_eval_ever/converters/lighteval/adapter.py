"""Adapter for converting lighteval output to every_eval_ever format."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.adapter import (
    AdapterMetadata,
    BaseEvaluationAdapter,
    SupportedLibrary,
)
from every_eval_ever.converters.common.utils import get_current_unix_timestamp
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceMetadata,
    SourceType,
    StandardError,
    Uncertainty,
)

from .utils import (
    KNOWN_METRIC_BOUNDS,
    PROVIDER_TO_INFERENCE_ENGINE,
    PROVIDER_TO_INFERENCE_PLATFORM,
    parse_model_name,
)


class LightEvalAdapter(BaseEvaluationAdapter):
    """Converts lighteval results to every_eval_ever format."""

    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)
        # Stores per-log metadata so callers can find sample files after transform.
        # Keyed by evaluation_id -> {"parent_dir": str, "task_name": str}
        self._eval_metadata = {}

    def get_eval_metadata(self, evaluation_id: str) -> Dict[str, Any]:
        """Return stored metadata for a given evaluation_id."""
        return self._eval_metadata.get(evaluation_id, {})

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='lighteval-adapter',
            version='0.1.0',
            supported_library_versions=['*'],
            description='Converts lighteval output to every_eval_ever format',
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.CUSTOM

    def _extract_model_info(
        self,
        raw_data: Dict[str, Any],
        metadata_args: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """Extract model information from lighteval results."""
        metadata_args = metadata_args or {}
        config_general = raw_data.get('config_general', {})
        model_config = config_general.get('model_config', {})

        model_name = config_general.get('model_name', '')
        provider, model_path = parse_model_name(model_name)

        developer = None
        if '/' in model_path:
            developer = model_path.split('/')[0]

        inference_platform = PROVIDER_TO_INFERENCE_PLATFORM.get(provider)

        # Determine inference engine name: CLI override > auto-detection from provider
        engine_name = metadata_args.get(
            'inference_engine'
        ) or PROVIDER_TO_INFERENCE_ENGINE.get(provider)
        engine_version = metadata_args.get('inference_engine_version')

        inference_engine = None
        if engine_name:
            inference_engine = InferenceEngine(
                name=engine_name, version=engine_version
            )

        additional = {}

        # Add relevant fields from model_config to additional_details
        # Exclude generation_parameters (handled separately in generation_config)
        # and api_key (sensitive)
        exclude_fields = {'model_name', 'generation_parameters', 'api_key'}

        for key, value in model_config.items():
            if key not in exclude_fields and value is not None:
                # Convert value to appropriate string representation
                if isinstance(value, (str, int, float, bool)):
                    additional[key] = str(value)
                elif isinstance(value, dict):
                    additional[key] = json.dumps(value)
                elif isinstance(value, list):
                    additional[key] = json.dumps(value)

        # Add provider if detected from model_name
        if provider:
            additional['provider'] = provider

        return ModelInfo(
            name=model_path,
            id=model_path,
            developer=developer,
            inference_platform=inference_platform,
            inference_engine=inference_engine,
            additional_details=additional if additional else None,
        )

    def _get_tasks(self, raw_data: Dict[str, Any]) -> List[str]:
        """Get task names that have actual metric results.

        lighteval uses format like "aime25|0" for tasks.
        We extract the base task name without the suffix.
        """
        results = raw_data.get('results', {})
        tasks = []
        for task_key in results.keys():
            # Skip the 'all' aggregation key
            if task_key == 'all':
                continue
            tasks.append(task_key)
        return tasks

    def _build_source_data(self, task_config: Dict[str, Any], task_name: str):
        """Build source_data from task config."""
        hf_repo = task_config.get('hf_repo', '')
        dataset_name = task_config.get('name', task_name)

        if hf_repo:
            hf_subset = task_config.get('hf_subset')
            # Use evaluation_splits if available, otherwise use few_shots_split
            splits = task_config.get('evaluation_splits', [])
            hf_split = splits[0] if splits else None

            return SourceDataHf(
                dataset_name=dataset_name,
                source_type='hf_dataset',
                hf_repo=hf_repo,
                hf_split=hf_split,
                hf_subset=hf_subset if hf_subset != 'default' else None,
            )
        return SourceDataPrivate(
            dataset_name=dataset_name,
            source_type='other',
        )

    def _build_generation_config(
        self, raw_data: Dict[str, Any]
    ) -> Optional[GenerationConfig]:
        """Build generation config from lighteval config."""
        config_general = raw_data.get('config_general', {})
        model_config = config_general.get('model_config', {})
        gen_params = model_config.get('generation_parameters', {})

        if not gen_params:
            return None

        args = GenerationArgs(
            temperature=gen_params.get('temperature'),
            top_p=gen_params.get('top_p'),
            top_k=gen_params.get('top_k'),
            max_tokens=gen_params.get('max_new_tokens'),
        )

        additional = {}
        for k, v in gen_params.items():
            if k not in ('temperature', 'top_p', 'top_k', 'max_new_tokens'):
                if v is not None:  # Only include non-None values
                    additional[k] = json.dumps(v) if not isinstance(v, str) else v

        # Add num_fewshot from task config if available
        # Note: lighteval stores this per-task, but we're extracting from general config
        # Individual tasks may override this

        return GenerationConfig(
            generation_args=args,
            additional_details=additional if additional else None,
        )

    def _build_evaluation_results(
        self, raw_data: Dict[str, Any], task_key: str
    ) -> List[EvaluationResult]:
        """Build EvaluationResult list for a single task."""
        task_results = raw_data['results'][task_key]
        task_config = raw_data.get('config_tasks', {}).get(task_key, {})

        source_data = self._build_source_data(task_config, task_key)
        gen_config = self._build_generation_config(raw_data)

        # lighteval stores timestamps in config_general
        config_general = raw_data.get('config_general', {})
        eval_timestamp = config_general.get('start_time')
        if eval_timestamp is not None:
            eval_timestamp = str(int(eval_timestamp))

        # Extract base task name (remove |0 suffix)
        task_name = task_key.split('|')[0]

        results = []
        for metric_key, value in task_results.items():
            # Skip stderr keys
            if '_stderr' in metric_key:
                continue
            if not isinstance(value, (int, float)):
                continue

            # Extract metric name (e.g., "pass@k:k=1&n=1" -> "pass@k")
            metric_name = metric_key.split(':')[0] if ':' in metric_key else metric_key

            # Get stderr if available
            stderr_key = f'{metric_key}_stderr'
            stderr_val = task_results.get(stderr_key)

            # Determine if higher is better from task config
            is_higher_better = True
            metrics_config = task_config.get('metrics', [])
            for metric_cfg in metrics_config:
                if metric_cfg.get('metric_name') == metric_key:
                    is_higher_better = metric_cfg.get('higher_is_better', True)
                    break

            # Determine score bounds
            bounds = KNOWN_METRIC_BOUNDS.get(metric_name)
            if bounds:
                min_score = bounds[0]
                max_score = bounds[1]
            else:
                # Infer bounds for common metric patterns
                metric_lower = metric_name.lower()
                if any(
                    pattern in metric_lower
                    for pattern in ['acc', 'accuracy', 'precision', 'recall', 'f1', 'pass', 'avg']
                ):
                    # Accuracy-like metrics are typically in [0, 1]
                    min_score = 0.0
                    max_score = 1.0
                else:
                    # Default fallback for unknown continuous metrics
                    min_score = 0.0
                    max_score = 1.0

            metric_config = MetricConfig(
                evaluation_description=metric_key,
                lower_is_better=not is_higher_better,
                score_type=ScoreType.continuous,
                min_score=min_score,
                max_score=max_score,
            )

            uncertainty = None
            num_samples = task_config.get('effective_num_docs')
            if num_samples == -1:
                num_samples = None

            # Only use stderr_val if it's a valid number
            valid_stderr = (
                isinstance(stderr_val, (int, float)) and stderr_val is not None
            )
            if valid_stderr or num_samples:
                uncertainty = Uncertainty(
                    standard_error=(
                        StandardError(value=stderr_val, method='bootstrap')
                        if valid_stderr
                        else None
                    ),
                    num_samples=num_samples,
                )

            results.append(
                EvaluationResult(
                    evaluation_name=task_name,
                    source_data=source_data,
                    evaluation_timestamp=eval_timestamp,
                    metric_config=metric_config,
                    score_details=ScoreDetails(
                        score=value,
                        uncertainty=uncertainty,
                    ),
                    generation_config=gen_config,
                )
            )

        return results

    def _transform_single(
        self, raw_data: Dict[str, Any], metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        """Transform a single task's results into an EvaluationLog.

        Expects metadata_args to contain 'task_key' specifying which task.
        """
        task_key = metadata_args['task_key']
        model_info = self._extract_model_info(raw_data, metadata_args)

        retrieved_timestamp = get_current_unix_timestamp()
        config_general = raw_data.get('config_general', {})
        eval_timestamp = config_general.get('start_time')
        if eval_timestamp is not None:
            eval_timestamp = str(int(eval_timestamp))

        # Extract base task name for ID (remove |0 suffix)
        task_name = task_key.split('|')[0]
        evaluation_id = f'{task_name}/{model_info.id}/{retrieved_timestamp}'
        evaluation_results = self._build_evaluation_results(raw_data, task_key)

        evaluator_rel_str = metadata_args.get(
            'evaluator_relationship', 'first_party'
        )
        evaluator_relationship = EvaluatorRelationship(evaluator_rel_str)

        # Prefer CLI argument, then config value if meaningful
        cli_version = metadata_args.get('eval_library_version', 'unknown')
        config_version = str(config_general.get('lighteval_sha', 'unknown'))

        # Use CLI version if provided and not 'unknown', otherwise use config
        if cli_version != 'unknown':
            library_version = cli_version
        elif config_version not in ('?', 'unknown'):
            library_version = config_version
        else:
            library_version = 'unknown'

        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name', 'lighteval'),
            version=library_version,
        )

        source_metadata = SourceMetadata(
            source_name='lighteval',
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

        # Store metadata so callers can find sample files after transform
        self._eval_metadata[evaluation_id] = {
            'parent_dir': metadata_args.get('parent_eval_output_dir'),
            'task_key': task_key,
        }

        return EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=eval_timestamp,
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a lighteval results JSON file into EvaluationLogs.

        Returns one EvaluationLog per task in the results file.
        """
        file_path = Path(file_path)
        raw_data = self._load_file(file_path)
        tasks = self._get_tasks(raw_data)

        # Pass the parent directory so instance-level adapter can find samples files
        if 'parent_eval_output_dir' not in metadata_args:
            metadata_args = {
                **metadata_args,
                'parent_eval_output_dir': str(file_path.parent),
            }

        results = []
        for task_key in tasks:
            task_metadata = {**metadata_args, 'task_key': task_key}
            log = self._transform_single(raw_data, task_metadata)
            results.append(log)

        return results

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform all lighteval results files in a directory.

        Searches for results_*.json files recursively.
        """
        dir_path = Path(dir_path)
        results_files = sorted(dir_path.glob('**/results_*.json'))

        all_logs = []
        for results_file in results_files:
            logs = self.transform_from_file(results_file, metadata_args)
            all_logs.extend(logs)

        return all_logs
