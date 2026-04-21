"""Adapter for converting SWE-bench output to every_eval_ever format."""

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
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataPrivate,
    SourceMetadata,
    SourceType,
)

from .utils import get_repo_name, parse_instance_id


class SWEBenchAdapter(BaseEvaluationAdapter):
    """Converts SWE-bench results to every_eval_ever format."""

    def __init__(self, strict_validation: bool = True):
        super().__init__(strict_validation)

    @property
    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name='swebench-adapter',
            version='0.1.0',
            supported_library_versions=['*'],
            description='Converts SWE-bench output to every_eval_ever format',
        )

    @property
    def supported_library(self) -> SupportedLibrary:
        return SupportedLibrary.CUSTOM

    def _extract_model_info(
        self,
        raw_data: Dict[str, Any],
        metadata_args: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """Extract model information from metadata args.

        SWE-bench results don't contain model info, so we rely on CLI args.
        """
        metadata_args = metadata_args or {}

        model_name = metadata_args.get('model_name', 'unknown')
        model_id = metadata_args.get('model_id', model_name)

        developer = None
        if '/' in model_id:
            developer = model_id.split('/')[0]

        additional = {}
        if metadata_args.get('agent_system'):
            additional['agent_system'] = str(metadata_args['agent_system'])

        return ModelInfo(
            name=model_name,
            id=model_id,
            developer=developer,
            additional_details=additional if additional else None,
        )

    def _build_evaluation_results(
        self, raw_data: Dict[str, Any]
    ) -> List[EvaluationResult]:
        """Build EvaluationResult list from SWE-bench summary.

        Creates aggregate summary results containing ID lists and counts.
        """
        results = []

        # Get all instance lists from raw data
        resolved_ids = raw_data.get('resolved_ids', [])
        unresolved_ids = raw_data.get('unresolved_ids', [])
        error_ids = raw_data.get('error_ids', [])
        empty_patch_ids = raw_data.get('empty_patch_ids', [])
        completed_ids = raw_data.get('completed_ids', [])
        incomplete_ids = raw_data.get('incomplete_ids', [])
        submitted_ids = raw_data.get('submitted_ids', [])
        total_instances = raw_data.get('total_instances', 0)
        submitted_instances = raw_data.get('submitted_instances', 0)
        completed_instances = raw_data.get('completed_instances', 0)
        resolved_instances = raw_data.get('resolved_instances', 0)
        unresolved_instances = raw_data.get('unresolved_instances', 0)
        empty_patch_instances = raw_data.get('empty_patch_instances', 0)
        error_instances = raw_data.get('error_instances', 0)

        # Common source data for summary metrics
        summary_source_data = SourceDataPrivate(
            dataset_name='swe-bench',
            source_type='other',
        )

        # Helper function to create aggregate results with ID lists
        def create_aggregate_result(
            name: str,
            count: int,
            id_list: List[str],
        ) -> EvaluationResult:
            """Create an aggregate metric with embedded ID list."""
            # Store IDs as JSON array string in evaluation_description
            ids_json = json.dumps(sorted(id_list))

            return EvaluationResult(
                evaluation_name=f'swe-bench-summary/{name}',
                source_data=summary_source_data,
                evaluation_timestamp=None,
                metric_config=MetricConfig(
                    evaluation_description=ids_json,
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=float(total_instances) if total_instances > 0 else 1.0,
                ),
                score_details=ScoreDetails(
                    score=float(count),
                    uncertainty=None,
                ),
                generation_config=None,
            )

        # Add aggregate summary results with ID lists
        results.append(
            create_aggregate_result(
                'total_instances',
                total_instances,
                submitted_ids,
            )
        )

        results.append(
            create_aggregate_result(
                'submitted_instances',
                submitted_instances,
                submitted_ids,
            )
        )

        results.append(
            create_aggregate_result(
                'completed_instances',
                completed_instances,
                completed_ids,
            )
        )

        results.append(
            create_aggregate_result(
                'resolved_instances',
                resolved_instances,
                resolved_ids,
            )
        )

        results.append(
            create_aggregate_result(
                'unresolved_instances',
                unresolved_instances,
                unresolved_ids,
            )
        )

        results.append(
            create_aggregate_result(
                'error_instances',
                error_instances,
                error_ids,
            )
        )

        results.append(
            create_aggregate_result(
                'empty_patch_instances',
                empty_patch_instances,
                empty_patch_ids,
            )
        )

        if incomplete_ids:
            results.append(
                create_aggregate_result(
                    'incomplete_instances',
                    len(incomplete_ids),
                    incomplete_ids,
                )
            )

        return results

    def _transform_single(
        self, raw_data: Dict[str, Any], metadata_args: Dict[str, Any]
    ) -> EvaluationLog:
        """Transform SWE-bench summary into an EvaluationLog.

        Args:
            raw_data: SWE-bench evaluation.json summary file
            metadata_args: Metadata from CLI args
        """
        model_info = self._extract_model_info(raw_data, metadata_args)

        retrieved_timestamp = get_current_unix_timestamp()
        evaluation_id = f'swe-bench/{model_info.id}/{retrieved_timestamp}'
        evaluation_results = self._build_evaluation_results(raw_data)

        evaluator_rel_str = metadata_args.get(
            'evaluator_relationship', 'third_party'
        )
        evaluator_relationship = EvaluatorRelationship(evaluator_rel_str)

        # SWE-bench schema version if available
        schema_version_val = raw_data.get('schema_version')
        library_version = (
            str(schema_version_val) if schema_version_val else 'unknown'
        )

        eval_library = EvalLibrary(
            name=metadata_args.get('eval_library_name', 'swe-bench'),
            version=metadata_args.get('eval_library_version', library_version),
        )

        source_metadata = SourceMetadata(
            source_name='swe-bench',
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
            evaluation_timestamp=None,  # Not in SWE-bench summary
            source_metadata=source_metadata,
            eval_library=eval_library,
            model_info=model_info,
            evaluation_results=evaluation_results,
        )

    def transform_from_file(
        self, file_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform a SWE-bench evaluation.json file into an EvaluationLog.

        Returns a single-item list containing the EvaluationLog.
        """
        file_path = Path(file_path)
        raw_data = self._load_file(file_path)

        # Pass the parent directory for context
        if 'parent_eval_output_dir' not in metadata_args:
            metadata_args = {
                **metadata_args,
                'parent_eval_output_dir': str(file_path.parent),
            }

        log = self._transform_single(raw_data, metadata_args)
        return [log]

    def transform_from_directory(
        self, dir_path: Union[str, Path], metadata_args: Dict[str, Any]
    ) -> List[EvaluationLog]:
        """Transform all SWE-bench evaluation.json files in a directory.

        Searches for evaluation.json files recursively.
        """
        dir_path = Path(dir_path)
        eval_files = sorted(dir_path.glob('**/evaluation.json'))

        all_logs = []
        for eval_file in eval_files:
            logs = self.transform_from_file(eval_file, metadata_args)
            all_logs.extend(logs)

        return all_logs
