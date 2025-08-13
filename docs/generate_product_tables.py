#!/usr/bin/env python3
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dist_s1.data_models.output_models import ProductNameData


def extract_field_info(model_class: type) -> list[dict[str, str]]:
    """Extract field information from a Pydantic model."""
    fields = []

    for field_name, field_info in model_class.model_fields.items():
        # Skip private fields
        if field_name.startswith('_'):
            continue

        # Get field type
        field_type = str(field_info.annotation).replace("<class '", '').replace("'>", '')

        # Get default value
        if field_info.default is not None and str(field_info.default) != 'PydanticUndefined':
            default_value = str(field_info.default)
        else:
            default_value = 'No default'

        # Get description
        description = field_info.description or 'No description available'

        # Get required status
        required = field_info.default is None or str(field_info.default) == 'PydanticUndefined'

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
    """Generate ProductNameData documentation table."""
    docs_dir = Path(__file__).parent

    # Generate ProductNameData table
    product_name_fields = extract_field_info(ProductNameData)
    product_name_md = generate_markdown_table(product_name_fields, 'ProductNameData Fields')

    # Write to file
    out_path = docs_dir / 'product_models.md'
    with Path.open(out_path, 'w', encoding='utf-8') as f:
        f.write(product_name_md)

    print('Product model documentation table generated successfully!')


if __name__ == '__main__':
    main()
