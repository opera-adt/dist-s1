from pathlib import Path

base_dir = Path("tests/test_data/products_without_confirmation_cropped__chile-fire_2024")

for product_dir in base_dir.iterdir():
    if product_dir.is_dir():
        browse_name = f"{product_dir.name}_BROWSE.png"
        browse_path = product_dir / browse_name
        browse_path.touch()
