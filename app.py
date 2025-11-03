import streamlit as st
import pandas as pd
import io, json, os, textwrap, re
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
import warnings
import requests
from urllib.parse import urlparse, parse_qs
import time
import streamlit.components.v1 as components
from utils import COMMON_DATE_FORMATS,format_date,SYSTEM_PROMPT,AZURE_OPENAI_KEY,AZURE_OPENAI_ENDPOINT,AZURE_OPENAI_DEPLOYMENT,AZURE_OPENAI_API_VERSION


components.html(
    """
    <script>
        // Detect the actual Streamlit sidebar key
        const key = Object.keys(window.localStorage).find(k => k.startsWith("stSidebarCollapsed-"));
        if (!key) {
            console.log("❌ No sidebar key found");
        } else {
            const state = window.localStorage.getItem(key);
            console.log("🔎 Sidebar state", key, "=", state);

            // ✅ Prevent multiple reload loops
            const guard = sessionStorage.getItem("sidebar_fix_done");

            if (state === "true" && !guard) {
                console.log("🛠️ Sidebar collapsed → expanding now...");
                window.localStorage.setItem(key, "false");
                sessionStorage.setItem("sidebar_fix_done", "1");
                setTimeout(() => window.location.reload(), 300);
            } else {
                console.log("✅ Sidebar already open or fix already applied");
                sessionStorage.removeItem("sidebar_fix_done");
            }
        }
    </script>
    """,
    height=0,
)




st.set_page_config(page_title="BetaCode AI", page_icon="🤖", layout="wide")

# Hide Streamlit default menu and footer
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)



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



if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    st.error("Please set all required Azure OpenAI environment variables before running.")
    st.stop()

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# ---------------------
# Utility Functions
# ---------------------

def call_openai_to_get_action(system_prompt, user_prompt, columns):
    """
    Calls the model and ensures valid JSON for data actions.
    Handles greetings and thank-you messages gracefully.
    """

    # Lowercase and strip punctuation for easier matching
    cleaned = re.sub(r"[^\w\s]", "", user_prompt.lower()).strip()

    # Friendly/greeting detection
    greeting_keywords = ["hi", "hello", "hey", "who are you", "help", "what can you"]
    thank_keywords = ["thank", "thanks", "thank you"]

    # ✅ Smarter thank-you detection
    if (
        any(k in cleaned for k in thank_keywords)
        and not any(word in cleaned for word in ["data", "contact", "show", "list", "created", "meeting", "report", "count"])
        and cleaned.endswith(("thank you", "thanks", "thankyou"))
    ):
        friendly_response = (
            "😊 You're welcome! Glad I could help. "
            "Let me know if you’d like me to explore or analyze anything else in your CRM data."
        )
        return json.dumps({"action": "text", "text": friendly_response})

    # Friendly greeting detection (only if short and not data-related)
    if len(cleaned.split()) <= 6 and any(k in cleaned for k in greeting_keywords):
        friendly_response = (
            "👋 Hi! I'm **Zoya AI** — your **CRM Assistant** powered by BetaCode.\n\n"
            "I can help you explore and analyze your CRM data. "
            "For example, you can ask me:\n\n"
            "- 🔍 *Filter contacts created in October 2025*\n"
            "- 📊 *Give a bar graph for months vs. count of contacts created in that month*\n"
            "- 📈 *Show how many hot contacts each CR Manager has added each month till date*\n"
            "- 🧩 *Give the number of hot contacts*\n\n"
            "Just type what you need — I’ll take care of the rest!"
        )
        return json.dumps({"action": "text", "text": friendly_response})

    # Normal data-related prompt
    full_user_prompt = f"""
    The available columns are: {columns}.
    Now respond in JSON only based on the user's request: "{user_prompt}".
    """

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ],
    )

    raw_text = response.choices[0].message.content.strip()

    # Validate JSON output
    try:
        json.loads(raw_text)
        return raw_text
    except Exception:
        safe_json = {"action": "error", "message": f"Invalid model output: {raw_text}"}
        return json.dumps(safe_json)

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

    # Handle friendly or greeting responses
    if act in ["friendly", "text"]:
        return {"type": "text", "text": action.get("text", "Hi there!")}
    if act == "friendly":
        return {"type": "text", "text": action.get("text", "Hi there!")}

    
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
        group_cols = action.get("columns") or action.get("column")

        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # ✅ Auto-detect fallback for "who"/"person"/"employee" queries
        if not group_cols:
            possible_cols = [c for c in df.columns if "CREATED_BY" in c.upper() or "EMPLOYEE" in c.upper() or "NAME" in c.upper()]
            if possible_cols:
                group_cols = [possible_cols[0]]  # pick the best match automatically
            else:
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
                # allow non-value aggs like 'count', 'size', or 'max count per group'
                if not values_col or values_col not in df2.columns:
                    if str(agg).lower() in ["count", "size", "nunique"]:
                        grouped = df2.groupby(group_cols).size().reset_index(name="Count")
                    else:
                        return {"type": "error", "message": f"groupby with agg '{agg}' requires 'values' column."}
                else:
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
                return {"Data not found"}
        except Exception as e:
            return {"Data not found"}

        return {"type": "text", "text": f"{agg.title()} of '{column}' = {result_value}"}

    return {"Data not found"}

# ---------------------
# Streamlit UI
# ---------------------

st.markdown(
    """
    <div style="font-family: Poppins, sans-serif; color: #fff; text-align: left; display: inline-block; width: fit-content;">
        <h3 style="
            font-weight: 500;
            font-size: 40px;
            margin: 0;
            margin-bottom: -20px;
            margin-right: -35px;
        ">
            Ask <span style="color:#1abc9c;">Zoya AI</span>
        </h3>
        <div style="
            text-align: right;
            font-size: 12px;
            font-weight: 400;
            color: rgba(255,255,255,0.8);
            margin-bottom: 15px;
        ">
            Powered by BetaCode
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
        }

        div.stDownloadButton > button {
            background: linear-gradient(90deg, #313866, #504099);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 15px;
            font-weight: 500;
            width: 100%;
            transition: all 0.3s ease-in-out;
        }

        div.stDownloadButton > button:hover {
            # background: linear-gradient(90deg, #5b5fc7, #6f73f1);
            # transform: scale(1.03);
            box-shadow: 0 0 10px rgba(90, 90, 255, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <style>
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
    # Dynamically reset uploader when analyse_clicked is True
    if st.session_state.get("analyse_clicked"):
        uploader_key = f"uploader_reset_{int(time.time())}"  # force new key
    else:
        uploader_key = "file_uploader"

    uploaded = st.file_uploader(
        "Choose file (.xlsx, .xls, .csv)",
        type=["xlsx","xls","csv"],
        key=uploader_key
    )

    if "last_uploaded_name" not in st.session_state:
        st.session_state["last_uploaded_name"] = None
    if "rerun_triggered" not in st.session_state:
        st.session_state["rerun_triggered"] = False
    if uploaded is not None:
        st.session_state["analyse_clicked"] = False
        if uploaded.name != st.session_state["last_uploaded_name"]:
            st.session_state["last_uploaded_name"] = uploaded.name
            st.session_state["rerun_triggered"] = False
        file_bytes = uploaded.read()
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        
        st.session_state["df"] = df  # ✅ Save to session for later access

        st.success(f"Uploaded file with {len(df)} rows and {len(df.columns)} columns.")
        # st.dataframe(df.head(10))
    else:
        if st.session_state["last_uploaded_name"] is not None and not st.session_state["rerun_triggered"]:
            st.session_state["last_uploaded_name"] = None
            st.session_state["rerun_triggered"] = True
            st.session_state["df"] = None  # ✅ Save to session for later access

            st.rerun()

    st.markdown("### Or")

   

    if "loading" not in st.session_state:
        st.session_state.loading = False

    # CSS — perfectly mimic Streamlit’s st.button look
    st.markdown("""
    <style>
    .loading-btn {
        background-color: #2B2B36;
        color: white;
        border: none;
        border-radius: 0.5rem;
        border: 1px solid #53545D;
        padding: 0.5rem 1rem;
        font-family: inherit;
        font-weight: 500;
        font-size: 0.9rem;
        cursor: not-allowed;
        opacity: 0.85;
        display: inline-flex;
        align-items: center;
        justify-content: center;
                width: 96%;
    }
    .loading-btn:hover {
        background-color: #343337;
        opacity: 0.85;
    }
    .loading-spinner {
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-top: 2px solid white;
        border-radius: 50%;
        width: 14px;
        height: 14px;
        margin-left: 8px;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Button + loader logic
    def connect_to_crm():
        if not st.session_state.loading:
            if st.button("🧩 Connect my CRM data source"):
                st.session_state.loading = True
                query_params = st.query_params
                AUTH_ID = query_params.get("AUTH_ID", [None])[0] if isinstance(query_params.get("AUTH_ID"), list) else query_params.get("AUTH_ID")

                if not AUTH_ID:
                    st.error("❌  No Data found")
                    st.session_state.loading = False  # optional: reset state
                    return
                else:
                    st.session_state["auth_id"] = AUTH_ID
                    st.session_state["analyse_clicked"] = True
                st.rerun()
        else:
            st.markdown('<button class="loading-btn">Connecting<span class="loading-spinner"></span></button>', unsafe_allow_html=True)
            time.sleep(5)
            st.session_state.loading = False
            st.session_state["analyse_clicked"] = True
            st.rerun()
    connect_to_crm()
    st.markdown("---")
    file_path="sample_data.xlsx"
    st.markdown("### 📁 Download Center")
    st.download_button(
        label="⬇️ Download Sample Data",
        data=open(file_path, "rb"),
        file_name="sample_data.xlsx",
        mime="application/octet-stream"
    )

if st.session_state.get("analyse_clicked"):
    try:
        uploaded = None  
        
        AUTH_ID = st.session_state["auth_id"]
        CRM_SERVER_URL = os.getenv("CRM_SERVER_URL")

        # ✅ Construct full API URL
        api_url = f"{CRM_SERVER_URL}/apis/sharepoint/contactDataGet"

        # ✅ Send POST with JSON body
        payload = {"AUTH_ID": AUTH_ID,"PAGE_NUMBER":-1}
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            data = response.json()

            # ✅ Always try to take data["value"] if it exists
            if isinstance(data, dict) and "value" in data:
                employee_name = data["employee_name"]
                data = data["value"]

            # ✅ Now check if it's a list
            if isinstance(data, list):
                # st.success(f"👋 Hi {employee_name}! Fetched {len(data)} contacts from CRM data source.".format(**data[0] if data else {}))
                st.markdown(f"""
<div style="
    font-family: 'Poppins', sans-serif;
    background-color: #193929;  /* translucent parrot green */
    color: #5DE488;  /* bright success green text */
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 10px;
">
👋 Hi <b>{employee_name}</b>! Fetched <b>{len(data)}</b> contacts from CRM data source.
</div>
""", unsafe_allow_html=True)

                remove_fields = ["_id", "__v","IS_DELETED", "isRemoved", "ID", "crManager","industry"]

        # ✅ Define rename mapping
                rename_map = {
                    "name0": "Name",
                    "type": "Type",
                    "designation": "Designation",
                    "email": "Email",
                    "department": "Department",
                    "mobile": "Mobile",
                    "company": "Company",
                    "landline": "Landline",
                    "linkedIn": "LinkedIn",
                    "address": "Address",
                    "level": "Level",
                    "state": "State",
                    "city": "City",
                    "CREATED_DATE": "Created Date",
                    "CREATED_BY": "Created By",
                    "engagementStatus": "Engagement Status",
                    "CREATED_BY_NAME": "Created By",
                    "crManagerName": "CR Manager",
                    "crManagerNameEmail": "CR Manager Email",
                    # add more mappings as needed
                }
                engagementStatus = [
                    {"label": "Cold", "value": "1"},
                    {"label": "Warm", "value": "2"},
                    {"label": "Hot", "value": "3"},
                    {"label": "Signed", "value": "4"},
                    {"label": "Dropped", "value": "5"},
                ]
                # make a quick lookup dict
                engagement_map = {item["value"]: item["label"] for item in engagementStatus}

                cleaned_data = []
                for item in data:
                    item["CREATED_DATE"] = format_date(item["CREATED_DATE"])

                    if "engagementStatus" in item:
                        val = str(item["engagementStatus"])  # convert to string just in case
                        item["engagementStatus"] = engagement_map.get(val, val) 
                    # ✅ Remove unwanted fields
                    cleaned = {k: v for k, v in item.items() if k not in remove_fields}

                    # ✅ Rename fields according to mapping
                    cleaned = {rename_map.get(k, k): v for k, v in cleaned.items()}

                    cleaned_data.append(cleaned)
                df = pd.DataFrame(cleaned_data)
                st.session_state["df"] = df
                # st.success(f"Fetched {len(df)} rows")
            else:
                st.error("Data not found")
                st.stop()
        else:
            st.error(f"Something went wrong while fetching data")
            st.stop()

        # ✅ Store df in session for later use
        st.session_state["df"] = df

    except Exception as e:
        pass


if not uploaded and "df" not in st.session_state:
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
            To get started, upload your contact list Excel file or work on your CRM data by clicking 'Connect my CRM data source' and enter your questions to analyze your data.<br><br>
            <b>Example Prompts:</b><br><br>
            • Give the list of contacts created in October 2025<br><br>
            • How many contacts were created in August 2025<br><br>
            • Give a bar graph for months vs count of contacts created in that month<br><br>
            • Give multi-line charts showing how many contacts each CR Manager has added in each month till date. Lines should be color-coded for each CR Manager.<br>
            <br>
            👈 Ready to explore? Download the sample contact file and start prompting!
            
        </div>
        """,
        unsafe_allow_html=True
    )


    st.stop()

# Load dataset
# Load dataset
if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            # ✅ Decode bytes → text, then pass to pandas
            stringio = io.StringIO(uploaded.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
        else:
            df = pd.read_excel(uploaded)

        # ✅ Drop completely empty rows
        df = df.dropna(how="all")

        # ✅ Drop rows that contain only spaces or blanks
        df = df[~df.apply(lambda row: all(str(x).strip() == "" for x in row), axis=1)]

        # ✅ Reset index
        df = df.reset_index(drop=True)

    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

# ---------------------
# Main Display Section
# ---------------------

# Retrieve the DataFrame safely from session state
df = st.session_state.get("df")

if df is not None:
    st.markdown(
        f"✅ **Loaded:** `{uploaded.name if uploaded else 'BetaCode CRM Data'}` — "
        f"**{len(df)} rows × {len(df.columns)} cols**"
    )

    # 🧠 Show full scrollable table (not truncated)
    st.dataframe(
        df,
        use_container_width=True,
        height=400  # adjust for screen
    )

    # 🧠 Input area for user prompt
    st.markdown("---")
    # user_prompt = st.text_area(
    #     "Ask anything about your data:",
    #     placeholder="e.g. Show total contacts created per month by CR Manager",
    #     height=120,
    # )

    # if st.button("Run"):
    #     with st.spinner("Thinking..."):
    #         result_text = call_openai_to_get_action(SYSTEM_PROMPT, user_prompt, df.columns)
    #         try:
    #             action = safe_load_json(result_text)
    #             output = run_action(df, action)
    #         except Exception as e:
    #             st.error(f"❌ Failed to parse model output: {e}")
    #             st.text(result_text)
    #             st.stop()

    #         if output["type"] == "error":
    #             st.error(output["message"])
    #         elif output["type"] == "text":
    #             st.success(output["text"])
    #         elif output["type"] == "table":
    #             st.dataframe(output["data"], use_container_width=True, height=600)
    #         elif output["type"] == "chart":
    #             st.altair_chart(output["chart"], use_container_width=True)
else:
    st.session_state.history = []
    st.info("👈 Upload a file or click **Connect my CRM data source** to get started.")
    
# if uploaded is not None:
#     st.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} cols")
# else:
#     st.success(f"Loaded fetched data — {df.shape[0]} rows × {df.shape[1]} cols")
# st.dataframe(df.head(10))

# Keep prompt box visible persistently (chat-like)
# --- Ensure history persists ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Display conversation history first ---
for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["user"])
    with st.chat_message("assistant"):
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
                response = result["text"]

                # Clean up if contains "Count of"
                if "Count of" in response:
                    import re
                    match = re.search(r"= ?(\d+)", response)
                    if match:
                        response = f"Count = {match.group(1)}"

                st.write(response)

        except Exception as e:
            st.warning(f"⚠️ Hello! Data could not be found")

# --- Create placeholder for spinner just above input ---
spinner_placeholder = st.empty()

# --- Chat input ---
prompt = st.chat_input("Ask something about your data (e.g. 'filter contacts created in October 2025')")

if prompt:
    with spinner_placeholder.container():  # spinner appears at the bottom
        with st.spinner("Thinking..."):
            try:
                raw_action_text = call_openai_to_get_action(SYSTEM_PROMPT, prompt, list(df.columns))
                st.session_state.history.append({"user": prompt, "model": raw_action_text})

                # --- Detect non-JSON output (like "Hi, who are you") ---
                if not raw_action_text.strip().startswith("{"):
                    result = {"type": "text", "text": raw_action_text}
                else:
                    try:
                        action = safe_load_json(raw_action_text)
                        result = run_action(df, action)
                    except Exception as e:
                        result = {"type": "error", "message": str(e)}

            except Exception as e:
                st.error(f"❌ Error while generating response: {e}")

    # trigger rerun so new message shows immediately
    st.rerun()
