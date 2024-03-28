#!/bin/bash
pip install auditwheel
python3 tool/ci_build.py "$@"