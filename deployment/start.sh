#!/bin/bash
set -e
echo 'Starting Streamlit app...'
python3 -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
