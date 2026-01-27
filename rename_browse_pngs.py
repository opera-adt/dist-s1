"""Rename PNG files in tests/test_data/ from <OPERA_ID>.png to <OPERA_ID>_BROWSE.png"""

from pathlib import Path


def rename_png_files(base_dir: str = "tests/test_data/") -> None:
    """
    Rename all .png files to add _BROWSE suffix before extension.

    Parameters
    ----------
    base_dir : str
        Base directory to search for PNG files
    """
    base_path = Path(base_dir)
    png_files = list(base_path.rglob("*v0.1.png"))

    for png_file in png_files:
        if "_BROWSE.png" in png_file.name:
            continue

        new_name = png_file.stem + "_BROWSE.png"
        new_path = png_file.parent / new_name
        png_file.rename(new_path)


if __name__ == "__main__":
    rename_png_files()
