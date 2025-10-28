import streamlit as st
import pandas as pd
import io, json, os, textwrap
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
import warnings

COMMON_DATE_FORMATS = [
    "%d/%m/%Y %I:%M:%S %p",  # 24/10/2025 8:54:46 PM
    "%d/%m/%Y %H:%M:%S",     # 24/10/2025 20:54:46
    "%d/%m/%Y",              # 24/10/2025
    "%Y-%m-%d %H:%M:%S",     # 2025-10-24 20:54:46
    "%Y-%m-%d",              # 2025-10-24
]

def parse_datetime_col(df, col):
    """
    Robustly parse df[col] into datetimes.
    Tries common explicit formats first (to avoid pandas warning).
    Falls back to pd.to_datetime(..., dayfirst=True) only if needed.
    Returns a Series of dtype datetime64[ns].
    """
    series = df[col].astype(str).replace({'nan': None, 'None': None})
    best_parsed = None
    best_valid = -1

    for fmt in COMMON_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            valid = parsed.notna().sum()
            # pick the format that yields the most valid parses
            if valid > best_valid:
                best_valid = valid
                best_parsed = parsed
            # if this format parsed the majority, accept immediately
            if valid / max(1, len(series.dropna())) > 0.75:
                return parsed
        except Exception:
            continue

    # If one of the explicit formats was best, return it
    if best_parsed is not None and best_valid > 0:
        return best_parsed

    # Last resort: fallback to flexible parsing (dateutil) with dayfirst
    # Suppress the pandas warning here as we know it's the fallback
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.to_datetime(series, errors="coerce", dayfirst=True)

# ---------------------
# Config
# ---------------------
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    st.error("Please set all required Azure OpenAI environment variables before running.")
    st.stop()

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

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

# ---------------------
# Utility Functions
# ---------------------

def call_openai_to_get_action(system_prompt, user_prompt, df_columns):
    context = f"Available columns: {', '.join(df_columns)}"
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\n{user_prompt}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

def safe_load_json(text):
    try:
        return json.loads(text)
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise


def apply_filters(df, filters):
    if not filters:
        return df

    out = df.copy()

    for f in filters:
        col, op, val = f.get("column"), f.get("op"), f.get("value")

        if col not in out.columns:
            raise KeyError(f"column missing: {col}")

        # 🗓️ Convert to datetime if it's a date column
        if "Date" in col:
            out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)

        # ✅ Handle month-based filtering like {"op": "month", "value": "2025-10"}
        if op == "month" and "Date" in col:
            try:
                year, month = map(int, val.split("-"))
                out = out[
                    (out[col].dt.year == year) &
                    (out[col].dt.month == month)
                ]
                continue
            except Exception:
                pass

        # ✅ Handle other date comparisons (>, >=, <, <=)
        if op in [">", ">=", "<", "<="] and "Date" in col:
            try:
                val_dt = pd.to_datetime(val, errors="coerce")
                out = out.query(f"`{col}` {op} @val_dt")
                continue
            except Exception:
                pass

        # 🧠 Convert to string for text filtering
        col_series = out[col].astype(str)

        # ✅ Equal match (case-insensitive, supports partial word matches)
        if op == "==":
            # First, try exact match
            matched = out[col_series.str.lower() == str(val).lower()]
            # If no exact match found, fallback to partial contains
            if matched.empty:
                matched = out[col_series.str.contains(str(val), case=False, na=False)]
            out = matched

        elif op == "!=":
            out = out[~col_series.str.lower().eq(str(val).lower())]

        # ✅ Substring match
        elif op == "contains":
            out = out[col_series.str.contains(str(val), case=False, na=False)]

        # ✅ "in" operator for lists or comma-separated strings
        elif op == "in":
            if isinstance(val, str):
                val_list = [v.strip() for v in val.split(",")]
            else:
                val_list = val
            val_list = [str(v).lower() for v in val_list]
            out = out[col_series.str.lower().isin(val_list)]

        else:
            try:
                out = out.query(f"`{col}` {op} @val")
            except Exception:
                pass

    return out

def run_action(df, action):
    act = action.get("action")
    if act == "preview":
        return {"type":"table","data":df.head(10)}
    if act == "show_columns":
        return {"type":"text","text":", ".join(df.columns)}
    if act == "describe":
        return {"type":"table","data":df.describe(include='all').T}
    if act == "sort":
        col, order = action.get("column"), action.get("order","desc")
        if col not in df.columns: return {"type":"error","message":f"column missing: {col}"}
        return {"type":"table","data":df.sort_values(col, ascending=(order=="asc")).reset_index(drop=True)}
    if act == "filter":
        df2 = apply_filters(df, action.get("filters", []))
        return {"type":"table","data":df2.reset_index(drop=True)}
       # -------------------------
   
    # -------------------------
    if act == "plot":
        plot_type = action.get("plot_type", "line")
        x_col = action.get("x")
        y_col = action.get("y", "count")
        filters = action.get("filters", [])
        date_range = action.get("date_range", None)

        # detect explicit grouping instruction ONLY (do not auto-fallback)
        raw_group = action.get("group_by") or action.get("groupby") or action.get("group")
        group_by_requested = None
        if isinstance(raw_group, list) and len(raw_group) > 0:
            group_by_requested = raw_group[0]
        elif isinstance(raw_group, str):
            group_by_requested = raw_group

        df2 = df.copy()

        # Apply filters early
        if filters:
            df2 = apply_filters(df2, filters)

        # Validate x_col presence
        if not x_col or x_col not in df2.columns:
            return {"type": "error", "message": f"plot requires valid 'x' column (e.g. 'Created Date'). Got: {x_col}"}

        # Parse x_col into datetime robustly using your helper
        df2[x_col] = parse_datetime_col(df2, x_col)
        df2 = df2.dropna(subset=[x_col])
        df2["Month-Year"] = df2[x_col].dt.to_period("M").astype(str)

        # Apply date_range if present (respect dayfirst via parse_datetime_col)
        if date_range:
            dr_col = date_range.get("column", x_col)
            start = pd.to_datetime(date_range.get("start"), errors="coerce")
            end = pd.to_datetime(date_range.get("end"), errors="coerce")
            if dr_col in df2.columns:
                df2[dr_col] = parse_datetime_col(df2, dr_col)
                df2 = df2[(df2[dr_col] >= start) & (df2[dr_col] <= end)]
                df2 = df2.dropna(subset=[dr_col])

        # -------------------------
        # CASE A: Explicit group_by requested -> produce multi-line (one line per group)
        # -------------------------
        if group_by_requested:
            # Resolve group_by: allow case-insensitive match and strip spaces
            candidate = None
            cols_map = {c.strip().lower(): c for c in df2.columns}
            if group_by_requested.strip().lower() in cols_map:
                candidate = cols_map[group_by_requested.strip().lower()]
            else:
                # try direct exact match fallback (rare)
                if group_by_requested in df2.columns:
                    candidate = group_by_requested

            if not candidate:
                return {"type": "error", "message": f"Group-by column '{group_by_requested}' not found in data."}

            group_by = candidate
            # normalize group_by values
            df2[group_by] = df2[group_by].astype(str).str.strip()

            # Use size() => counts rows regardless of empty Name fields
            pivot = df2.groupby(["Month-Year", group_by]).size().reset_index(name="Count")

            # Build full months range between min and max month
            try:
                min_month = pd.Period(pivot["Month-Year"].min(), freq="M")
                max_month = pd.Period(pivot["Month-Year"].max(), freq="M")
                months = pd.period_range(start=min_month, end=max_month, freq="M").astype(str)
            except Exception:
                months = sorted(pivot["Month-Year"].unique())

            groups = sorted(pivot[group_by].unique())

            # Full grid so every manager has an entry for every month (fill missing with 0)
            full = pd.MultiIndex.from_product([months, groups], names=["Month-Year", group_by]).to_frame(index=False)
            merged = full.merge(pivot, on=["Month-Year", group_by], how="left").fillna(0)
            merged["Count"] = merged["Count"].astype(int)
            merged["Month-Year"] = pd.Categorical(merged["Month-Year"], categories=months, ordered=True)

            # Build Altair multi-line chart
            base = alt.Chart(merged).encode(
                x=alt.X("Month-Year:N", sort=months, title="Month"),
                y=alt.Y("Count:Q", title="Contacts Created"),
                color=alt.Color(f"{group_by}:N", title=group_by),
                tooltip=["Month-Year", group_by, "Count"]
            )

            if plot_type == "bar":
                chart = base.mark_bar().properties(width=900, height=450)
            else:
                chart = base.mark_line(point=True).properties(width=900, height=450)

            return {"type": "chart", "chart": chart}

        # -------------------------
        # CASE B: No grouping requested -> total per month (count rows)
        # -------------------------
        grouped = df2.groupby("Month-Year").size().reset_index(name="Count").sort_values("Month-Year")
        months = sorted(grouped["Month-Year"].unique())
        grouped["Month-Year"] = pd.Categorical(grouped["Month-Year"], categories=months, ordered=True)

        base = alt.Chart(grouped).encode(
            x=alt.X("Month-Year:N", sort=months, title="Month"),
            y=alt.Y("Count:Q", title="Total Contacts"),
            tooltip=["Month-Year", "Count"]
        )

        if plot_type == "bar":
            chart = base.mark_bar().properties(width=900, height=450)
        else:
            chart = base.mark_line(point=True).properties(width=900, height=450)

        return {"type": "chart", "chart": chart}

    if act == "groupby":
        # Accept both 'column' (string) or 'columns' (list)
        group_cols = action.get("columns") or action.get("column")
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        if not group_cols:
            return {"type": "error", "message": "groupby requires 'columns' or 'column'."}

        # Agg and optional values (what to aggregate)
        agg = action.get("agg", "count")
        values_col = action.get("values") or action.get("value")  # support "values" or "value"

        # Start with a copy and apply filters first (so grouping uses filtered rows)
        df2 = df.copy()
        if action.get("filters"):
            df2 = apply_filters(df2, action.get("filters", []))

        # Apply date_range if present (consistent with aggregate)
        date_range = action.get("date_range", None)
        if date_range:
            dr_col = date_range.get("column")
            start = pd.to_datetime(date_range.get("start"), errors="coerce")
            end = pd.to_datetime(date_range.get("end"), errors="coerce")
            if dr_col in df2.columns:
                df2[dr_col] = pd.to_datetime(df2[dr_col], errors="coerce", dayfirst=True)
                df2 = df2[(df2[dr_col] >= start) & (df2[dr_col] <= end)]
                df2 = df2.dropna(subset=[dr_col])

        # If the grouping column is a date-like field and the user expects month grouping
        if len(group_cols) == 1 and "Date" in group_cols[0]:
            col0 = group_cols[0]
            df2[col0] = pd.to_datetime(df2[col0], errors="coerce", dayfirst=True)
            df2["Month-Year"] = df2[col0].dt.to_period("M").astype(str)
            grouped = df2.groupby("Month-Year").size().reset_index(name="Count")
            return {"type": "table", "data": grouped}

        # Perform grouping+aggregation
        try:
            if str(agg).lower() == "count":
                # count rows per group (if values_col provided, count non-null values there)
                if values_col and values_col in df2.columns:
                    grouped = df2.groupby(group_cols)[values_col].count().reset_index(name="Count")
                else:
                    grouped = df2.groupby(group_cols).size().reset_index(name="Count")
            else:
                # other aggregations (require values_col)
                if not values_col or values_col not in df2.columns:
                    return {"type": "error", "message": "groupby with agg requires 'values' (column to aggregate)."}
                grouped = df2.groupby(group_cols)[values_col].agg(agg).reset_index()
            return {"type": "table", "data": grouped}
        except Exception as e:
            return {"type": "error", "message": f"groupby failed: {e}"}

    # ✅ NEW: handle "aggregate"
    elif act == "aggregate":
        column = action.get("column")
        agg = action.get("agg", "count").lower()
        filters = action.get("filters", [])
        df2 = apply_filters(df, filters)

        # ✅ Apply date_range if present (was missing earlier)
        date_range = action.get("date_range", None)
        if date_range:
            colname = date_range.get("column")
            start = pd.to_datetime(date_range.get("start"), errors="coerce")
            end = pd.to_datetime(date_range.get("end"), errors="coerce")

            if colname in df2.columns:
                df2[colname] = pd.to_datetime(df2[colname], errors="coerce", dayfirst=True)
                df2 = df2[(df2[colname] >= start) & (df2[colname] <= end)]

        if column not in df2.columns:
            return {"type": "error", "message": f"column missing: {column}"}

        # Perform aggregation
        try:
            if agg == "count":
                if not column or column == "*" or "name" in column.lower():
                    result_value = len(df2.dropna(how="all"))
                else:
                    result_value = df2[column].count()
            elif agg == "sum":
                result_value = df2[column].sum()
            elif agg in ["mean", "avg"]:
                result_value = df2[column].mean()
            elif agg == "min":
                result_value = df2[column].min()
            elif agg == "max":
                result_value = df2[column].max()
            else:
                return {"type": "error", "message": f"unsupported aggregation: {agg}"}
        except Exception as e:
            return {"type": "error", "message": f"aggregation failed: {e}"}

        return {"type": "text", "text": f"{agg.title()} of '{column}' = {result_value}"}

    return {"type":"error","message":f"unsupported action: {act}"}



# ---------------------
# Streamlit UI
# ---------------------
st.markdown(
    """
    <h3 style="
        font-family: Poppins, sans-serif;
        font-weight: 500;
        font-size: 40px;
        color: #fff;
        margin-bottom: 10px;
    ">
        Ask <span style="color:#1abc9c;">BetaCode AI</span>
    </h3>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        """
        <style>
        /* Sidebar branding: "BetaCode" text + small green 'BETA' chip */
        [data-testid="stSidebar"]::before {
            content: "BetaCode";
            position: absolute;
            top: 10px;
            left: 15px;
            font-size: 35px;
            font-weight: bold;
            color: #fff;
            font-family: Poppins;
        }

        /* Add 'BETA' chip */
      [data-testid="stSidebar"]::after {
    content: "BETA";
    position: absolute;
    top: 16px;
    left: 160px; /* adjust for alignment */
    background-color: rgba(26, 188, 156, 0.25); /* translucent base green (#1abc9c) */
    color: #1abc9c; /* same base color for text */
    font-size: 12px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    border: 1px solid rgba(26, 188, 156, 0.8); /* darker border from same base */
    font-family: Poppins, sans-serif;
    backdrop-filter: blur(3px);
}






        /* Add margin below to prevent overlap */
        [data-testid="stSidebar"] section:first-child {
            margin-top: 45px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### Upload an Excel / CSV file")
    uploaded = st.file_uploader("Choose file (.xlsx, .xls, .csv)", type=["xlsx","xls","csv"])
    st.markdown("---")
    # st.markdown("### Model / Settings")
    # st.text_input("Model", value=AZURE_OPENAI_DEPLOYMENT, key="model_field")
    # st.markdown("System prompt (used internally)")
    # if st.button("Show system prompt"):
    #     st.code(SYSTEM_PROMPT, language="text")

if not uploaded:
    st.markdown(
        """
        <style>
        .custom-info-box {
            background-color: #0e1117; /* dark background */
            color: white !important;
            border-left: 4px solid #1abc9c; /* optional accent border */
            padding: 15px;
            border-radius: 8px;
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
        }
        .custom-info-box b {
            color: #1abc9c;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="custom-info-box">
            To get started, upload your contact list Excel file and enter your questions to analyze your data.<br><br>
            <b>Example Prompts:</b><br><br>
            • Give the list of contacts created in October 2025<br><br>
            • How many contacts were created in August 2025<br><br>
            • Give a bar graph for months vs count of contacts created in that month<br><br>
            • Give line charts showing how many contacts each CR Manager has added in each month till date. Lines should be color-coded for each CR Manager.<br>
            
        </div>
        """,
        unsafe_allow_html=True
    )


    st.stop()

# Load dataset
# Load dataset
try:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
        
    # ✅ Drop completely empty rows (rows with all NaN/blank values)
    df = df.dropna(how="all")

    # ✅ Drop rows that are technically empty (like only spaces)
    df = df[~df.apply(lambda row: all(str(x).strip() == "" for x in row), axis=1)]

    # ✅ Reset index after cleaning
    df = df.reset_index(drop=True)

except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()


st.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
st.dataframe(df.head(10))

# Keep prompt box visible persistently (chat-like)
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input("Ask something about your data (e.g. 'filter contacts created in October 2025')")

if prompt:
    with st.spinner("Thinking..."):
        raw_action_text = call_openai_to_get_action(SYSTEM_PROMPT, prompt, list(df.columns))
        st.session_state.history.append({"user": prompt, "model": raw_action_text})
        try:
            action = safe_load_json(raw_action_text)
            result = run_action(df, action)
        except Exception as e:
            result = {"type": "error", "message": str(e)}

# Display conversation
for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["user"])
    with st.chat_message("assistant"):
        # st.code(h["model"], language="json")

        try:
            action = safe_load_json(h["model"])
            result = run_action(df, action)
            if result["type"] == "error":
                st.error(result["message"])
            elif result["type"] == "table":
                st.dataframe(result["data"])
            elif result["type"] == "chart":
                st.altair_chart(result["chart"], use_container_width=True)
            elif result["type"] == "text":
                st.write(result["text"])
        except:
            st.warning("Could not parse model output.")
