#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Any


# Add src to path to import dist_s1 modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dist_s1.data_models.algoconfig_model import AlgoConfigData
from dist_s1.data_models.defaults import *  # noqa: F403
from dist_s1.data_models.runconfig_model import RunConfigData


def is_field_required(field_info: Any) -> bool:  # noqa: ANN401
    """Determine if a field is required based on its configuration."""
    # Check if field has a default value
    if field_info.default is not None and str(field_info.default) != 'PydanticUndefined':
        return False

    # Check if field is marked as required
    if hasattr(field_info, 'is_required') and field_info.is_required:
        return True

    # Check if field has a default in defaults.py
    field_name = field_info.name if hasattr(field_info, 'name') else None
    if field_name:
        default_var_name = f'DEFAULT_{field_name.upper()}'
        try:
            default_value = globals()[default_var_name]
            if default_value is not None:
                return False
        except KeyError:
            pass

    # Check if type hint indicates Optional (Union with None)
    type_hint = field_info.annotation
    if type_hint is not None:
        type_str = str(type_hint)
        if 'None' in type_str or 'Optional' in type_str:
            return False

    # Default to required if no default is found
    return True


def get_default_value(field_name: str, field_info: Any) -> str:  # noqa: ANN401
    """Get the default value for a field from defaults.py or field default."""
    # First check if field has a default value
    if field_info.default is not None and str(field_info.default) != 'PydanticUndefined':
        default_value = field_info.default
        if isinstance(default_value, (str, int, float, bool)):
            return str(default_value)
        elif isinstance(default_value, (list, tuple)):
            return str(default_value)
        elif isinstance(default_value, Path):
            return f'`{default_value}`'
        else:
            return str(default_value)

    # If no field default, check defaults.py
    default_var_name = f'DEFAULT_{field_name.upper()}'
    try:
        default_value = globals()[default_var_name]
        if default_value is None:
            return 'None'
        elif isinstance(default_value, (str, int, float, bool)):
            return str(default_value)
        elif isinstance(default_value, (list, tuple)):
            return str(default_value)
        elif isinstance(default_value, Path):
            return f'`{default_value}`'
        else:
            return str(default_value)
    except KeyError:
        return 'No default'


def format_type_hint(type_hint: Any) -> str:  # noqa: ANN401
    """Format type hints for display in documentation."""
    if type_hint is None:
        return 'Any'

    # Handle string representation of types
    type_str = str(type_hint)

    # Clean up common type representations
    type_str = type_str.replace('pathlib._local.Path', 'Path')
    type_str = type_str.replace('dist_s1.data_models.output_models.DistS1ProductDirectory', 'DistS1ProductDirectory')
    type_str = type_str.replace('dist_s1.data_models.algoconfig_model.AlgoConfigData', 'AlgoConfigData')
    type_str = type_str.replace("<class '", '').replace("'>", '')

    # Handle Union types (including Optional)
    if '|' in type_str and 'Union' not in type_str:
        # This is already a union type string
        return type_str
    elif 'Union[' in type_str:
        # Convert Union to pipe notation
        union_content = type_str.replace('Union[', '').replace(']', '')
        types = [t.strip() for t in union_content.split(',')]
        return ' | '.join(types)

    return type_str


def extract_field_info(model_class: type) -> list[dict[str, str]]:
    """Extract field information from a Pydantic model."""
    fields = []

    for field_name, field_info in model_class.model_fields.items():
        # Skip private fields
        if field_name.startswith('_'):
            continue

        # Get field type
        field_type = format_type_hint(field_info.annotation)

        # Get default value
        default_value = get_default_value(field_name, field_info)

        # Get description
        description = field_info.description or 'No description available'

        # Get required status
        required = is_field_required(field_info)

        fields.append(
            {
                'name': field_name,
                'type': field_type,
                'default': default_value,
                'description': description,
                'required': required,
            }
        )

    return fields


def generate_markdown_table(fields: list[dict[str, str]], title: str) -> str:
    """Generate a markdown table from field information."""
    markdown = f'## {title}\n\n'
    markdown += '| Attribute | Type | Default | Required | Description |\n'
    markdown += '|-----------|------|---------|----------|-------------|\n'

    for field in fields:
        # Escape pipe characters in description
        description = field['description'].replace('|', '\\|')
        required_text = 'Yes' if field['required'] else 'No'
        markdown += (
            f'| `{field["name"]}` | `{field["type"]}` | {field["default"]} | {required_text} | {description} |\n'
        )

    return markdown


def main() -> None:
    """Generate API documentation tables."""
    docs_dir = Path(__file__).parent

    # Generate RunConfigData table
    runconfig_fields = extract_field_info(RunConfigData)
    runconfig_md = generate_markdown_table(runconfig_fields, 'RunConfigData')

    # Generate AlgoConfigData table
    algoconfig_fields = extract_field_info(AlgoConfigData)
    algoconfig_md = generate_markdown_table(algoconfig_fields, 'AlgoConfigData')

    # Write to files
    out_path = docs_dir / 'config' / 'runconfig.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(out_path, 'w', encoding='utf-8') as f:
        f.write(runconfig_md)

    out_path = docs_dir / 'config' / 'algoconfig.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(out_path, 'w', encoding='utf-8') as f:
        f.write(algoconfig_md)

    print('Config documentation tables generated successfully!')


if __name__ == '__main__':
    main()
