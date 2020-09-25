#!/usr/bin/env bash
nohup python -u app.py  --app_port 8003 --db_host 10.21.23.210 --db_database document-ml-test --db_user document-ml-test  > test.log 2>&1 &