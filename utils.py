from datetime import datetime
import io, json, os, textwrap, re
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
COMMON_DATE_FORMATS = [
    "%d/%m/%Y %I:%M:%S %p",  # 24/10/2025 8:54:46 PM
    "%d/%m/%Y %H:%M:%S",     # 24/10/2025 20:54:46
    "%d/%m/%Y",              # 24/10/2025
    "%Y-%m-%d %H:%M:%S",     # 2025-10-24 20:54:46
    "%Y-%m-%d",              # 2025-10-24
]

def format_date(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y %I:%M:%S %p")
    except Exception:
        return iso_str  # return as-is if not a valid date

SYSTEM_PROMPT = textwrap.dedent("""
You are a helpful assistant that converts a user's natural-language request about a spreadsheet
into a JSON "action" that a program will run. The JSON must be valid and *only* JSON (no prose).

Rules:
- The JSON must be a single object with an "action" key.
- Allowed actions: "show_columns", "preview", "sort", "filter", "aggregate", "plot", "describe", "topk", "groupby".
- Common fields:
    - "column": column name (string)
    - "columns": list of column names
    - "order": "asc" or "desc"
    - "plot_type": one of ["bar","line","scatter","hist","box"]
    - "x", "y": column names for plotting
    - "agg": for aggregate function ("sum","mean","count","min","max")
    - "filters": list of { "column": ..., "op": "==|!=|>|<|>=|<=|contains", "value": ... }
    - "k": integer for top-k
    - "date_range": { "column": "...", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }

Examples:

1) Sort:
{"action":"sort","column":"Revenue","order":"desc"}

2) Bar plot of product vs sales:
{"action":"plot","plot_type":"bar","x":"Product","y":"Sales"}

3) Top 10 by Sales:
{"action":"topk","column":"Sales","k":10,"order":"desc"}

4) Filter by date:
{"action":"filter","filters":[{"column":"Created Date","op":"month","value":"2025-10"}]}

Important:
- If the column the user mentions does not exist in the dataset, return an action with "action":"error" and "message":"column missing: <name>".
- Dates should be formatted as "YYYY-MM-DD" in date_range and filters if the user gives date ranges.
- Always output JSON ONLY (no explanation).
""").strip()

