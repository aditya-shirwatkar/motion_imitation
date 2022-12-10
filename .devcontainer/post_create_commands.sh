#!/bin/bash

./../session/post_start/sync_deps.py
# pip install -r requirements.txt

python setup.py install --user

pip install -e .
