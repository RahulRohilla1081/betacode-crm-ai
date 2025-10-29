### step 1

python3 -m venv venv

## step 2

- Mac

source venv/bin/activate

- Windows
  venv\Scripts\activate

### step 3

pip install -r requirements.txt

### step 4

streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false --server.port 8501

# Or run as module

python -m streamlit run app.py
