#!/bin/bash
set -e
echo "Starting Streamlit server..."
# Use absolute path for safety and disable telemetry prompts
python3 -m streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.enableCORS false \
    --server.enableXsrfProtection false
