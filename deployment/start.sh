
#!/bin/bash
set -x # Print commands and their arguments as they are executed.

python -m streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --logger.level debug
