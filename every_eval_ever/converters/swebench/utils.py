"""Utility functions for SWE-bench adapter."""

from typing import Dict


def parse_instance_id(instance_id: str) -> Dict[str, str]:
    """Parse SWE-bench instance ID into components.

    Args:
        instance_id: Instance ID like "django__django-10914" or "sympy__sympy-24909"

    Returns:
        Dictionary with 'repo_owner', 'repo_name', and 'issue_number'
    """
    parts = instance_id.split('__')
    if len(parts) != 2:
        return {
            'repo_owner': 'unknown',
            'repo_name': 'unknown',
            'issue_number': instance_id,
        }

    repo_owner = parts[0]
    repo_and_issue = parts[1].rsplit('-', 1)

    if len(repo_and_issue) == 2:
        repo_name = repo_and_issue[0]
        issue_number = repo_and_issue[1]
    else:
        repo_name = parts[1]
        issue_number = 'unknown'

    return {
        'repo_owner': repo_owner,
        'repo_name': repo_name,
        'issue_number': issue_number,
    }


def get_repo_name(instance_id: str) -> str:
    """Get the repository name from instance ID.

    Args:
        instance_id: Instance ID like "django__django-10914"

    Returns:
        Repository name like "django/django"
    """
    parts = parse_instance_id(instance_id)
    return f"{parts['repo_owner']}/{parts['repo_name']}"
