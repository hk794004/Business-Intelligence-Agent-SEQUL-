import sys
import io
import streamlit as st
import os
import re
import pyodbc
import json
import time
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

try:
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer") and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"]       = "1"   

load_dotenv()

def ensure_str(val) -> str:
    """Convert anything to a clean Unicode str — never raises."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    try:
        return str(val)
    except Exception:
        return repr(val)

@st.cache_resource
def get_connection(driver, server, database, uid, pwd):
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        f"TrustServerCertificate=yes;"
        f"MARS_Connection=yes;"
    )
    return pyodbc.connect(conn_str, autocommit=True)


def fetch_schema(conn) -> dict:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.name AS schema_name,
            t.name AS table_name,
            c.name AS column_name,
            tp.name AS data_type,
            c.is_nullable
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        JOIN sys.columns c ON t.object_id = c.object_id
        JOIN sys.types tp ON c.user_type_id = tp.user_type_id
        ORDER BY s.name, t.name, c.column_id
    """)
    schema: dict = {}
    for row in cursor.fetchall():
        table_key = f"{row.schema_name}.{row.table_name}"
        nullable  = "NULL" if row.is_nullable else "NOT NULL"
        col_def   = f"{row.column_name} {row.data_type} {nullable}"
        schema.setdefault(table_key, []).append(col_def)
    return schema


def schema_to_prompt_text(schema: dict, server_label: str) -> str:
    lines = [f"\n### {server_label} Tables\n"]
    for table, cols in schema.items():
        lines.append(f"TABLE: {table}")
        for col in cols:
            lines.append(f"    {col}")
        lines.append("")
    return "\n".join(lines)


def rows_to_dicts(cursor) -> list:
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def inject_top_limit(sql: str, limit: int = 100000) -> str:
    stripped = sql.strip()
    if re.search(r'\bTOP\b', stripped, re.IGNORECASE):
        return stripped
    if re.match(r'^\s*WITH\b', stripped, re.IGNORECASE):
        return stripped
    return re.sub(r'\bSELECT\b', f'SELECT TOP {limit}', stripped, count=1, flags=re.IGNORECASE)

DECISION_TEMPLATE = """
You are a highly advanced Senior SQL Business Intelligence Agent router. Your primary responsibility is to construct clean, optimized SQL 'SELECT' statements based strictly on the provided schema definitions.

Strict Operational Instructions:
1. SECURITY: You are strictly restricted to SQL 'SELECT' statements. Do NOT write or allow any DML/DDL queries (INSERT, UPDATE, DELETE, DROP, ALTER).

2. OUT OF SCOPE & ZERO HALLUCINATION: If the user asks a question that cannot be resolved using the available tables/schema below, you MUST immediately set "out_of_scope": true. Do NOT hallucinate or use external knowledge.

3. VISUALIZATION ROUTING:
   - Set "needs_chart": true when ANY of these apply:
       * User mentions rankings, top N, bottom N, comparisons by category/city/region/product
       * User asks to "dikhao", "chart banao", "plot karo", "graph dikhao", "visualize"
       * The result will have a categorical column + one or more numeric metric columns (multi-row result)
       * The question involves sales, revenue, orders, customers grouped by any dimension
   - Set "needs_chart": false ONLY for single-value scalar results

4. ANALYSIS ROUTING:
   - Set "needs_analysis": true when ANY of these apply:
       * User asks for top/bottom rankings, comparisons, trends, performance review
       * The result will have multiple rows with business metrics
       * User uses words like "analysis", "insight", "compare", "best", "worst", "dikhao", "batao", "performance"
   - Set "needs_analysis": false ONLY for single scalar lookups

5. TIME SERIES ROUTING:
   - Set "needs_timeseries": true when result contains date/year/month columns with numeric metrics

6. MODEL SELECTION:
   - "fast": single table, simple aggregates, basic counts
   - "advanced": multi-table joins, window functions, complex aggregations

Available database schema:
{schema_text}

Output ONLY a raw valid JSON object (no markdown, no backticks):
{{
  "sql": "YOUR_OPTIMIZED_SQL_QUERY_HERE",
  "needs_chart": true or false,
  "needs_timeseries": true or false,
  "needs_analysis": true or false,
  "recommended_model_tier": "fast" or "advanced",
  "out_of_scope": false
}}
"""

ANALYSIS_TEMPLATE = """
You are an expert Executive Business Consultant and Senior Data Scientist acting as a friendly, professional AI analytics companion. Your core mandate is to deliver an intelligent diagnostic evaluation derived strictly from the fetched dataset results.

User's Input Question: {user_question}
Mapped Dataset Metric Layout:
{df_markdown}

Analytical Framework:
- DESCRIPTIVE & TREND VALUE: State exactly what the figures show clearly without repeating rigid templates.
- SEGMENTATION & BENCHMARKS: Detail performance metrics, segment groupings, or performance standouts captured within the data.
- OPERATIONAL FORECASTING: Outline forward-looking operational growth vectors or velocity drops if temporal elements are mapped.

Strict Communication Rules:
1. OUT OF SCOPE: If the data cannot logically answer the request, return exactly: "Out Of Scope Context Not Found in the Provided Document".
2. LANGUAGE INTELLIGENCE: If the user writes in Roman Urdu (e.g. 'mujhe batayo', 'karo', 'dikhao'), output the entire analysis in that same Roman Urdu style. Keep technical metric names in English.

Deliver your review concisely (max 3-4 precise bullet points).
"""

def intelligent_agent_decision(user_question, schema_text, chat_history, api_key) -> dict:
    llm = ChatGroq(api_key=api_key, model=Models, temperature=0)
    messages = [SystemMessage(content=DECISION_TEMPLATE.format(schema_text=schema_text))]

    for msg in chat_history:
        try:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=ensure_str(msg.get("content", ""))))
            elif msg["role"] == "assistant":
                content_str = ensure_str(msg.get("content", ""))
                if msg.get("sql"):
                    content_str += f"\nGenerated SQL: {ensure_str(msg['sql'])}"
                messages.append(AIMessage(content=content_str))
        except Exception:
            continue   

    messages.append(HumanMessage(content=ensure_str(user_question)))

    response = llm.invoke(messages)
    text = ensure_str(response.content).strip()

    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "sql": text, "needs_chart": False, "needs_timeseries": False,
            "needs_analysis": False, "recommended_model_tier": "fast", "out_of_scope": False
        }

def generate_chart_logic(df, api_key, model_name, language_instruction, user_question) -> str:
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
    columns_info = df.dtypes.to_dict()
    total_rows   = len(df)

    prompt = f"""
You are a senior data visualization engineer. Generate a single, clean Plotly Express chart.

USER'S ORIGINAL QUESTION (source of truth):
\"\"\"{user_question}\"\"\"

DataFrame info:
- Columns & dtypes: {columns_info}
- Total rows available: {total_rows}

The DataFrame is already available as 'df'. Assign the final figure to 'fig'.

CRITICAL RULE 1 — ROW COUNT:
Extract the EXACT number requested (e.g. "top 20", "top 5").
Use df.head(N) or df.tail(N). If no number stated, use ALL rows.

CRITICAL RULE 2 — CHART TYPE:
- Rankings/comparisons → horizontal bar (px.bar orientation='h')
- Time trends → line chart (px.line)
- Proportions → pie chart (px.pie)
- Correlation → px.scatter
- Multiple metrics → grouped bar (barmode='group')

CRITICAL RULE 3 — SORTING:
- Ranking charts: sort descending by primary metric BEFORE slicing.
- Time series: sort ascending by date.

CRITICAL RULE 4 — DATA LABELS:
Always show data labels. For px.bar use text= then:
fig.update_traces(texttemplate='%{{text:,.0f}}', textposition='inside')

CRITICAL RULE 5 — YEAR/ID COLUMNS:
Cast year/id/calendar columns to str for discrete rendering.

CRITICAL RULE 6 — MULTIPLE METRICS:
If multiple numeric columns exist, pass as list to y= with barmode='group'.

LANGUAGE RULE FOR CHART TITLE:
{language_instruction}
Create a dynamic title from the user's question. Follow the language rule strictly.

OUTPUT RULE:
Output ONLY pure executable Python code — no markdown, no backticks, no comments.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    code = ensure_str(response.content).strip()
    if code.startswith("```python"):
        code = code.split("```python")[1].split("```")[0].strip()
    elif code.startswith("```"):
        code = code.split("```")[1].split("```")[0].strip()
    return code

def type_text(placeholder, text: str, delay: float = 0.018):
    """Stream text word-by-word with a blinking cursor."""
    words   = ensure_str(text).split(" ")
    current = ""
    for word in words:
        current += word + " "
        placeholder.markdown(current + "▌")
        time.sleep(delay)
    placeholder.markdown(current.strip())

st.set_page_config(page_title="SQL AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Adaptive SQL Intelligence Agent")
st.caption("An intelligent agent that dynamically decides whether to write queries, create charts, or provide strategic forecasts based on your question.")

st.divider()

if "schema"      not in st.session_state: st.session_state.schema      = {}
if "schema_text" not in st.session_state: st.session_state.schema_text = ""
if "messages"    not in st.session_state: st.session_state.messages    = []

with st.sidebar:
    st.header("⚙️ Control")

    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    if groq_api_key:
        st.success("🔗 GROQ API KEY is Connected & Running...")
    else:
        st.error("GROQ API KEY is Missing...")
        st.stop()

    Models = st.selectbox(
        "Select Models",
        [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ]
    )

    st.divider()
    st.subheader("🗄️ SQL Server Connection")

    db_driver   = st.text_input("DRIVER",         value=os.getenv("DRIVER",   "{ODBC Driver 17 for SQL Server}"))
    db_server   = st.text_input("SERVER",         value=os.getenv("SERVER",   ""))
    db_database = st.text_input("DATABASE",       value=os.getenv("DATABASE", ""))
    db_uid      = st.text_input("UID (User ID)",  value=os.getenv("UID",      ""))
    db_pwd      = st.text_input("PWD (Password)", type="password", value=os.getenv("PWD", ""))

    st.divider()

    if st.button("🔄 Load Schema", use_container_width=True):
        if not groq_api_key:
            st.error("🔑 Please enter your Groq API Key...")
        elif not db_server or not db_database:
            st.error("⚠️ Please fill in SERVER and DATABASE fields...")
        else:
            with st.spinner("Fetching schema..."):
                try:
                    conn = get_connection(db_driver, db_server, db_database, db_uid, db_pwd)
                    st.session_state.schema      = fetch_schema(conn)
                    st.session_state.schema_text = schema_to_prompt_text(
                        st.session_state.schema, db_database
                    )
                    st.success(f"{len(st.session_state.schema)} Tables Fetched")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.schema:
        st.divider()
        st.subheader("📋 Database Schema Details")
        for table_name, columns in st.session_state.schema.items():
            with st.expander(f"📁 {table_name}", expanded=False):
                st.markdown("**Columns & Types:**")
                for col in columns:
                    st.caption(f"🔹 {col}")

def render_assistant_elements(message, animate: bool = False):
    if "info" in message:
        st.info(message["info"])
        return

    if "sql" in message and message["sql"]:
        with st.expander("📄 Generated SQL Query", expanded=False):
            display_sql = re.sub(
                r'\bTOP\s+100000\b', 'TOP 100', message["sql"], flags=re.IGNORECASE
            )
            st.code(display_sql, language="sql")

    if "df" in message and message["df"] is not None:
        df = message["df"]
        st.markdown("***💡 Data Tables Results***")
        st.dataframe(df.head(100), use_container_width=True)

        if message.get("show_ts") and message.get("has_ts"):
            try:
                df_ts    = df.copy()
                date_col = message["date_col"]
                val_col  = message["val_col"]
                if df_ts[date_col].dtype in ['int64', 'float64']:
                    df_ts[date_col] = df_ts[date_col].astype(str)
                df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
                df_ts = df_ts.dropna(subset=[date_col]).sort_values(by=date_col)
                fig_ts = px.line(df_ts, x=date_col, y=val_col,
                                 title="📈 Time Series Trend & Moving Average", markers=True)
                fig_ts.add_scatter(
                    x=df_ts[date_col],
                    y=df_ts[val_col].rolling(window=2, min_periods=1).mean(),
                    name="Moving Avg"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception:
                pass

        if message.get("show_chart") and message.get("chart_code"):
            try:
                df_chart = df.copy()
                for col in df_chart.columns:
                    if any(k in col.lower() for k in ('year', 'id', 'calendar')):
                        df_chart[col] = df_chart[col].astype(str)
                ldict = {"df": df_chart, "px": px}
                exec(message["chart_code"], {}, ldict)
                st.plotly_chart(ldict["fig"], use_container_width=True)
            except Exception:
                pass

    if message.get("show_analysis") and message.get("analysis"):
        st.markdown("***🚀 Strategic Insights (Full Dataset Analysis)***")
        if animate:
            placeholder = st.empty()
            type_text(placeholder, message["analysis"])
        else:
            st.markdown(message["analysis"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.write(message["content"])
        if message["role"] == "assistant":
            render_assistant_elements(message, animate=False)

user_query = st.chat_input("💬 Ask a question...")

if user_query:
    if not groq_api_key:
        st.warning("🔑 Please enter your Groq API Key...")
    elif not st.session_state.schema_text:
        st.warning("⚠️ Please Load The Schema First...")
    else:
        current_history = list(st.session_state.messages)

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            msg_data = {
                "role": "assistant", "content": f"Processed query: {user_query}",
                "sql": "", "df": None, "analysis": "", "chart_code": "",
                "has_ts": False, "show_chart": False, "show_ts": False, "show_analysis": False
            }

            try:
                language_instruction = (
                    "Strict Rule: Always analyze the user's input language style from the query. "
                    "If they write in Roman Urdu (e.g. 'karo', 'dikhao', 'mujhe'), "
                    "write chart titles and analysis in that same Roman Urdu style. "
                    "Keep technical metric names in English."
                )

                with st.status("🧠 Agent thinking...", expanded=False) as status:
                    decision = intelligent_agent_decision(
                        user_query,
                        st.session_state.schema_text,
                        current_history,
                        groq_api_key
                    )

                    if decision.get("out_of_scope"):
                        status.update(label="Out of Scope — Context Not Found", state="error")
                        msg_data["info"] = "Out Of Scope Context Not Found in the Provided Document"
                    else:
                        msg_data["sql"] = decision.get("sql", "")
                        active_model    = Models

                        conn   = get_connection(db_driver, db_server, db_database, db_uid, db_pwd)
                        cursor = conn.cursor()

                        executed_sql = inject_top_limit(msg_data["sql"], limit=100000)
                        cursor.execute(executed_sql)
                        data = rows_to_dicts(cursor)

                        if data:
                            df = pd.DataFrame(data)
                            msg_data["df"] = df

                            is_single_value = (df.shape == (1, 1))

                            if is_single_value:
                                msg_data["show_chart"]    = False
                                msg_data["show_ts"]       = False
                                msg_data["show_analysis"] = False
                            else:
                                msg_data["show_chart"]    = decision.get("needs_chart",      False)
                                msg_data["show_ts"]       = decision.get("needs_timeseries", False)
                                msg_data["show_analysis"] = decision.get("needs_analysis",   False)

                            if msg_data["show_ts"]:
                                date_cols    = [c for c in df.columns if any(k in c.lower() for k in ('date','year','month'))]
                                numeric_cols = [c for c in df.columns if df[c].dtype in ['float64','int64']]
                                if date_cols and numeric_cols:
                                    msg_data["has_ts"]   = True
                                    msg_data["date_col"] = date_cols[0]
                                    msg_data["val_col"]  = numeric_cols[0]

                            if msg_data["show_chart"]:
                                msg_data["chart_code"] = generate_chart_logic(
                                    df, groq_api_key, active_model,
                                    language_instruction, user_query
                                )

                            if msg_data["show_analysis"]:
                                df_markdown = (
                                    df.to_markdown()
                                    if len(df) <= 200
                                    else df.describe().to_markdown()
                                )
                                analysis_prompt = ANALYSIS_TEMPLATE.format(
                                    user_question=user_query,
                                    df_markdown=df_markdown
                                )
                                analysis_llm = ChatGroq(
                                    api_key=groq_api_key, model=active_model, temperature=0
                                )
                                response = analysis_llm.invoke(
                                    [HumanMessage(content=analysis_prompt)]
                                )
                                msg_data["analysis"] = ensure_str(response.content).strip()

                render_assistant_elements(msg_data, animate=True)
                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Error occurred: {e}")
