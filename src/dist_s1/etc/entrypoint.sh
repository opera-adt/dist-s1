#!/bin/bash --login
set -e
exec python -um dist-s1 run "$@"