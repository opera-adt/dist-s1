# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.2]

### Added

- Entrypoint for CLI to localize data via internet

### Changed

- CLI entrypoints now utilize `dist-s1 run_sas` and `dist-s1 run` rathern than just `dist-s1`. 
  - The `dist-s1 run_sas` is the primary entrypoint for Science Application Software (SAS) for SDS operations. 
  - The `dist-s1 run` is the simplified entrypoint for external users, allowing for the localization of data from publicly available data sources.

## [0.0.1]

### Added

- Initial internal release of the DIST-S1 project. Test github release workflow
