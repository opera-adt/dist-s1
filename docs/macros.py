"""MkDocs macros for generating dynamic documentation tables.

This module provides macros that generate documentation tables directly from
the source code, ensuring the documentation always reflects current values.
"""

import types
from pathlib import Path

from jinja2.environment import Environment
from pydantic.fields import FieldInfo

from dist_s1 import constants
from dist_s1.data_models import defaults
from dist_s1.data_models.algoconfig_model import AlgoConfigData
from dist_s1.data_models.output_models import ProductNameData
from dist_s1.data_models.runconfig_model import RunConfigData


def define_env(env: Environment) -> None:
    """Define the macro environment for MkDocs."""
    # Make modules available in templates
    env.variables['defaults'] = defaults
    env.variables['constants'] = constants

    # Make model classes available in templates
    env.variables['RunConfigData'] = RunConfigData
    env.variables['AlgoConfigData'] = AlgoConfigData
    env.variables['ProductNameData'] = ProductNameData

    def is_field_required(field_info: FieldInfo) -> bool:
        """Determine if a field is required based on its configuration."""
        # Check if field has a default value
        if field_info.default is not None and str(field_info.default) != 'PydanticUndefined':
            return False

        # Check if field is marked as required
        if hasattr(field_info, 'is_required') and field_info.is_required:
            return True

        # Check if field has a default in defaults.py
        field_name = field_info.alias or (field_info.field_info.alias if hasattr(field_info, 'field_info') else None)
        if not field_name and hasattr(field_info, 'name'):
            field_name = field_info.name

        if field_name:
            default_var_name = f'DEFAULT_{field_name.upper()}'
            if hasattr(defaults, default_var_name):
                default_value = getattr(defaults, default_var_name)
                if default_value is not None:
                    return False

        # Check if type hint indicates Optional (with None)
        type_hint = field_info.annotation
        if type_hint is not None:
            type_str = str(type_hint)
            if 'None' in type_str or 'Optional' in type_str:
                return False

        # Default to required if no default is found
        return True

    def get_default_value(field_name: str, field_info: FieldInfo) -> str:
        """Get the default value for a field from defaults.py or field default."""
        # First check if field has a default value
        if field_info.default is not None and str(field_info.default) != 'PydanticUndefined':
            default_value = field_info.default
            if isinstance(default_value, str | int | float | bool):
                return str(default_value)
            elif isinstance(default_value, list | tuple):
                return str(default_value)
            elif isinstance(default_value, Path):
                return f'`{default_value}`'
            else:
                return str(default_value)

        # If no field default, check defaults.py
        default_var_name = f'DEFAULT_{field_name.upper()}'
        if hasattr(defaults, default_var_name):
            default_value = getattr(defaults, default_var_name)
            if default_value is None:
                return 'None'
            elif isinstance(default_value, str | int | float | bool):
                return str(default_value)
            elif isinstance(default_value, list | tuple):
                return str(default_value)
            elif isinstance(default_value, Path):
                return f'`{default_value}`'
            else:
                return str(default_value)

        return 'No default'

    def format_type_hint(type_hint: type | types.GenericAlias | None) -> str:
        """Format type hints for display in documentation."""
        if type_hint is None:
            return 'Any'

        # Handle string representation of types
        type_str = str(type_hint)

        # Clean up common type representations
        type_str = type_str.replace('pathlib._local.Path', 'Path')
        type_str = type_str.replace(
            'dist_s1.data_models.output_models.DistS1ProductDirectory', 'DistS1ProductDirectory'
        )
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

    @env.macro
    def generate_config_table(model_class: type, title: str = None) -> str:
        """Generate a markdown table from a Pydantic model."""
        if title is None:
            title = model_class.__name__

        fields = extract_field_info(model_class)

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

    @env.macro
    def generate_constants_table(constant_dict: dict, title: str, description_col: str = 'Description') -> str:
        """Generate a markdown table from a dictionary of constants."""
        markdown = f'## {title}\n\n'
        markdown += f'| Key | Value | {description_col} |\n'
        markdown += f'|-----|-------|{"-" * len(description_col)}|\n'

        for key, value in constant_dict.items():
            # Handle different value types
            if isinstance(value, str):
                formatted_value = f'`"{value}"`'
            elif isinstance(value, int | float):
                formatted_value = f'`{value}`'
            elif value is None or (hasattr(value, '__name__') and value.__name__ == 'nan'):
                formatted_value = '`NaN`'
            else:
                formatted_value = f'`{value}`'

            # For layer tables, we might want descriptions from constants
            if hasattr(constants, 'TIF_LAYER_DESCRIPTIONS') and key in constants.TIF_LAYER_DESCRIPTIONS:
                desc = constants.TIF_LAYER_DESCRIPTIONS[key].replace('|', '\\|')
            else:
                desc = 'No description available'

            markdown += f'| `{key}` | {formatted_value} | {desc} |\n'

        return markdown

    @env.macro
    def generate_disturbance_labels_table() -> str:
        """Generate the disturbance labels table from constants."""
        markdown = '## Disturbance Labels\n\n'
        markdown += '| Label | Value | Description |\n'
        markdown += '|-------|-------|-------------|\n'

        for label, value in constants.DISTLABEL2VAL.items():
            # Convert underscores to spaces and title case for display
            display_label = label.replace('_', ' ').title()
            markdown += f'| {display_label} | `{value}` | {label.replace("_", " ").capitalize()} status |\n'

        return markdown

    @env.macro
    def get_default_value_macro(var_name: str) -> str:
        """Get a default value by variable name."""
        if hasattr(defaults, var_name):
            value = getattr(defaults, var_name)
            if isinstance(value, str | int | float | bool):
                return str(value)
            else:
                return f'`{value}`'
        return 'Not found'

    @env.macro
    def get_constant_value_macro(var_name: str) -> str:
        """Get a constant value by variable name."""
        if hasattr(constants, var_name):
            value = getattr(constants, var_name)
            if isinstance(value, str | int | float | bool):
                return str(value)
            else:
                return f'`{value}`'
        return 'Not found'
