#!/bin/bash --login
set -e
exec python -um dist_s1 run "$@"