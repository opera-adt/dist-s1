import netrc
import os
from pathlib import Path


def ensure_earthdata_credentials(
    username: str | None = None,
    password: str | None = None,
    host: str = 'urs.earthdata.nasa.gov',
) -> None:
    """Ensure Earthdata credentials are provided in ~/.netrc.

    Earthdata username and password may be provided by, in order of preference, one of:
       * `netrc_file`
       * `username` and `password`
       * `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables
    and will be written to the ~/.netrc file if it doesn't already exist.
    """
    if username is None:
        username = os.getenv('EARTHDATA_USERNAME')

    if password is None:
        password = os.getenv('EARTHDATA_PASSWORD')

    netrc_file = Path.home() / '.netrc'
    if not netrc_file.exists() and username and password:
        netrc_file.write_text(f'machine {host} login {username} password {password}')
        netrc_file.chmod(0o000600)

    try:
        dot_netrc = netrc.netrc(netrc_file)
        username, _, password = dot_netrc.authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError, TypeError):
        raise ValueError(
            f'Please provide valid Earthdata login credentials via {netrc_file}, '
            f'username and password options, or '
            f'the EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.'
        )
