"""CLI for converting SWE-bench output to every_eval_ever format."""

import argparse
import json
import sys
import uuid
from pathlib import Path

from .adapter import SWEBenchAdapter


def main():
    parser = argparse.ArgumentParser(
        description='Convert SWE-bench output to every_eval_ever format'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        required=True,
        help='Path to evaluation.json file or directory containing evaluation files',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for converted files',
    )
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='Model identifier (e.g. "openai/gpt-4", "anthropic/claude-3-sonnet")',
    )
    parser.add_argument(
        '--source_organization_name',
        type=str,
        default='',
        help='Name of the organization that ran the evaluation',
    )
    parser.add_argument(
        '--evaluator_relationship',
        type=str,
        default='third_party',
        choices=['first_party', 'third_party', 'collaborative', 'other'],
        help='Relationship of the evaluator to the model',
    )
    parser.add_argument(
        '--source_organization_url',
        type=str,
        default=None,
        help='URL of the source organization',
    )
    parser.add_argument(
        '--source_organization_logo_url',
        type=str,
        default=None,
        help='Logo of the source organization',
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default='SWE-bench',
        help='Name of the benchmark (e.g. "SWE-bench", "SWE-bench-lite")',
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default='princeton-nlp/SWE-bench',
        help='HuggingFace repository for the benchmark dataset',
    )
    parser.add_argument(
        '--eval_library_name',
        type=str,
        default='swebench',
        help='Name of the evaluation library',
    )
    parser.add_argument(
        '--eval_library_version',
        type=str,
        default='unknown',
        help='Version of the evaluation library',
    )

    args = parser.parse_args()

    adapter = SWEBenchAdapter()
    output_dir = Path(args.output_dir)

    metadata_args = {
        'model_id': args.model_id,
        'source_organization_name': args.source_organization_name,
        'evaluator_relationship': args.evaluator_relationship,
        'source_organization_url': args.source_organization_url,
        'source_organization_logo_url': args.source_organization_logo_url,
        'benchmark_name': args.benchmark_name,
        'hf_repo': args.hf_repo,
        'eval_library_name': args.eval_library_name,
        'eval_library_version': args.eval_library_version,
    }

    log_path = Path(args.log_path)

    if log_path.is_file():
        logs = adapter.transform_from_file(log_path, metadata_args)
    elif log_path.is_dir():
        logs = adapter.transform_from_directory(log_path, metadata_args)
    else:
        print(f'Error: {log_path} is not a file or directory', file=sys.stderr)
        sys.exit(1)

    for log in logs:
        # Organize as: output_dir/{evaluation_name}/{developer}/{model_name}/{uuid}.json
        if log.evaluation_results:
            eval_name = log.evaluation_results[0].evaluation_name
        else:
            eval_name = 'unknown'

        model_parts = log.model_info.id.split('/')
        if len(model_parts) >= 2:
            developer = model_parts[0]
            model_name = '/'.join(model_parts[1:])
        else:
            developer = 'unknown'
            model_name = log.model_info.id

        out_path = output_dir / eval_name / developer / model_name
        out_path.mkdir(parents=True, exist_ok=True)

        eval_uuid = str(uuid.uuid4())
        out_file = out_path / f'{eval_uuid}.json'

        with open(out_file, 'w') as f:
            json.dump(
                log.model_dump(mode='json', exclude_none=True), f, indent=2
            )

        print(f'  {out_file}')

    print(f'\nConverted {len(logs)} evaluation log(s).')


if __name__ == '__main__':
    main()
