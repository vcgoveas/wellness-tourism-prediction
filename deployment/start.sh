#!/bin/bash
set -e
python3 -u -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --logger.level debug
