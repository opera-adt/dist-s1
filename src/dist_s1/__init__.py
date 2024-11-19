import warnings
from importlib.metadata import PackageNotFoundError, version

from .input_data_model import RunConfigModel


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = None
    warnings.warn(
        'package is not installed!\n'
        'Install in editable/develop mode via (from the top of this repo):\n'
        '   python -m pip install -e .\n',
        RuntimeWarning,
    )


__all__ = ['RunConfigModel']
