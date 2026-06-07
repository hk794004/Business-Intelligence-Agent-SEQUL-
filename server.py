import streamlit as st
import os
import pyodbc
import json
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

MODELS_POOL = {
    "fast": "openai/gpt-oss-120b",         
    "advanced": "openai/gpt-oss-120b",     
    "default": "openai/gpt-oss-120b"   
}

@st.cache_resource
def get_connection(driver, server, database, uid, pwd):
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={uid};"
        f"PWD={pwd};"
        "TrustServerCertificate=yes;"
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
        nullable = "NULL" if row.is_nullable else "NOT NULL"
        col_def = f"{row.column_name} {row.data_type} {nullable}"
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

def rows_to_dicts(cursor) -> list[dict]:
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

SYSTEM_TEMPLATE = """
You are an advanced, brain-driven SQL Server Intelligence Agent. You don't just write SQL; you deeply evaluate the user's business intent, structurally map out analytics fields, and dynamically plan multi-dimensional diagnostic needs.

Strict Operational Rules:
1. SECURITY: Only generate highly optimized, clean SQL 'SELECT' statements. Never allow DML/DDL.
2. VOLUME: Always explicitly include 'TOP 100000' in your SQL query to pull sufficient raw records for pipeline background metrics and analytics calculations.
3. ADAPTIVE BRAIN DECISION MAKING:
   You must carefully inspect the schema and target columns to see if they support advanced analytical fields. Output a strictly valid JSON object.
   - If the user asks for patterns over time, cycles, or future trends, set needs_timeseries to true.
   - If the user is evaluating demographic metrics, customer types, regional breakdowns, or product tiers, dynamically identify this as a Segmentation opportunity and flag needs_chart and needs_analysis to true.
   - If the query requires summary statistics, group totals, counts, percentages, or high cardinality categorical overviews, interpret it as a Descriptive or Forecasting opportunity, ensuring downstream visualization modules trigger.
4. MODEL SELECTION: 
   - Choose "fast" if the question is straightforward, plain counts, simple metadata lookups, or narrow single-table requests.
   - Choose "advanced" if the business problem requires deep statistical forecasting logic, granular cohort segmentations, multi-table joins, or advanced analytical window functions.
5. LANGUAGE RESPONSE INTELLIGENCE:
   - Carefully read the user's linguistic style. Follow the exact script, style, and language used across the chat history.

Your output must be RAW JSON ONLY, matching this schema precisely (do not wrap in markdown code blocks or backticks):
{{
  "sql": "YOUR_SQL_QUERY_HERE",
  "needs_chart": true/false,
  "needs_timeseries": true/false,
  "needs_analysis": true/false,
  "recommended_model_tier": "fast" or "advanced",
  "out_of_scope": false
}}

Available database schema structures:
{schema_text}
"""

def intelligent_agent_decision(user_question, schema_text, chat_history, api_key) -> dict:
    llm = ChatGroq(api_key=api_key, model=MODELS_POOL["default"], temperature=0)
    
    messages = [SystemMessage(content=SYSTEM_TEMPLATE.format(schema_text=schema_text))]
    
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            content_str = msg.get("content", "")
            if msg.get("sql"):
                content_str += f"\nGenerated SQL: {msg['sql']}"
            messages.append(AIMessage(content=content_str))
            
    messages.append(HumanMessage(content=user_question))
    
    response = llm.invoke(messages)
    text = response.content.strip()
    
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
        
    try:
        return json.loads(text)
    except Exception:
        return {"sql": text, "needs_chart": True, "needs_timeseries": False, "needs_analysis": True, "recommended_model_tier": "advanced", "out_of_scope": False}

def generate_advanced_analysis_from_summary(summary_text: str, api_key: str, model_name: str, language_instruction: str, chat_history: list) -> str:
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
    
    history_context = ""
    if chat_history:
        history_context = "Previous Context / Conversations:\n" + "\n".join([f"{m['role']}: {m.get('content', '')}" for m in chat_history[-3:]])

    prompt = f"""
    You are an expert Data Scientist and Senior Data Analyst Agent. Analyze the Aggregated/Statistical Summary of the FULL dataset provided below.
    
    {history_context}

    CRITICAL ANALYSIS FRAMEWORK:
    - DESCRIPTIVE ANALYSIS: Summarize the current state, high-performing benchmarks, and key metric summaries.
    - SEGMENTATION ANALYSIS: If there are grouped categorical variables, analyze top-performing customer cohorts, regions, or product segments.
    - FORECASTING & TRENDS: If time elements exist, project forward-looking growth velocities, potential velocity drops, or future patterns.

    Provide actionable data-driven conclusions and deliver exactly 1 sharp strategic recommendation based on your deep findings.
    
    LANGUAGE RULE:
    {language_instruction}
    
    Summary Data:
    {summary_text}
    
    Keep it strictly professional, clear, and highly customized to the data structure. Max 3-4 bullet points. Do not return generic text templates.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def generate_advanced_analysis(df: pd.DataFrame, api_key: str, model_name: str, language_instruction: str, chat_history: list) -> str:
    if df.empty: return "No data available."
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
    df_markdown = df.to_markdown()
    
    history_context = ""
    if chat_history:
        history_context = "Previous Context / Conversations:\n" + "\n".join([f"{m['role']}: {m.get('content', '')}" for m in chat_history[-3:]])

    prompt = f"""
    You are an expert Executive Business Consultant and Analytics Agent. Provide an intelligent diagnostic review based EXACTLY on these mapped metrics:
    {df_markdown}
    
    {history_context}

    CRITICAL ANALYTICAL FOCUS:
    - Inspect the dimensions dynamically. If text-categories match, run a quick Segmentation performance analysis. If values span time intervals, outline a baseline Descriptive trend or operational Forecasting trajectory.
    - State clearly what the figures show without any fixed pre-written phrases.
    
    LANGUAGE RULE:
    {language_instruction}
    
    Keep it extremely precise and strategic. Max 3 actionable bullet points.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

def generate_chart_logic(df: pd.DataFrame, api_key: str, model_name: str, language_instruction: str) -> str:
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
    columns_info = df.dtypes.to_dict()
    prompt = f"""
    Based on these pandas DataFrame columns: {columns_info}
    Provide ONLY valid Python code using Plotly Express (px) to visualize the data.
    
    CRITICAL INSTRUCTIONS:
    - The DataFrame is already available as 'df'.
    - Assign the final plotly figure to a variable named 'fig'.
    - Never skip chart generation or write comments instead of code. You MUST generate an active chart layout.
    - If there are columns representing categories, years, or IDs (e.g., 'CalendarYear', 'Year', 'ID'), treat them as categorical strings so Plotly renders proper discrete elements.
    
    MULTIPLE METRICS & GROUPING RULE:
    - If there are MULTIPLE numeric metric columns alongside a time/categorical column, you MUST pass both metrics as a list to the y-axis parameter to create a grouped/side-by-side bar chart, or use `barmode='group'`. 
    
    DATA LABELS INSIDE RULE (STRICT MANDATE):
    - You MUST display data value labels clearly inside or directly on top of the visual elements (bars/lines/points). 
    - For bar charts (`px.bar`), pass the metric column name to the `text` parameter during creation, and then immediately call `fig.update_traces(texttemplate='%{{text:,.0f}}', textposition='inside')` to ensure numeric markings are perfectly readable.
    
    SORTING & CLEAN VISUALIZATION RULE:
    - If any column represents months, you MUST sort the DataFrame chronologically by month before passing it to px.
    - For non-numeric text/categorical columns, you MUST sort the DataFrame alphabetically (A to Z) using `df.sort_values()`.
    - Isolate exactly the TOP 10 rows based on the primary numeric column to maintain a clean view if clutter occurs.
    
    LANGUAGE RULE FOR CHART TITLE:
    {language_instruction}
    
    DYNAMIC TITLE RULE:
    - You MUST create a contextually dynamic title for the chart based on the selected columns. Adhere strictly to the LANGUAGE RULE above when creating the title string. Do NOT use generic or static English text for the title.
    
    - Output ONLY pure executable Python code. No markdown formatting, no backticks, no text explanations.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.strip()
    if code.startswith("```python"):
        code = code.split("```python")[1].split("```")[0].strip()
    elif code.startswith("```"):
        code = code.split("```")[1].split("```")[0].strip()
    return code

st.set_page_config(page_title="SQL AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Adaptive SQL Intelligence Agent")
st.caption("An intelligent agent that dynamically decides whether to write queries, create charts, or provide strategic forecasts based on your question.")

st.divider()

if "schema" not in st.session_state: st.session_state.schema = {}
if "schema_text" not in st.session_state: st.session_state.schema_text = ""
if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Control")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if groq_api_key:
        st.success("🔗 GROQ API KEY Connected...")
    
    st.divider()
    st.subheader("🗄️ SQL Server Connection")
    
    db_driver = st.text_input("DRIVER", value=os.getenv("DRIVER", "{ODBC Driver 17 for SQL Server}"))
    db_server = st.text_input("SERVER", value=os.getenv("SERVER", ""))
    db_database = st.text_input("DATABASE", value=os.getenv("DATABASE", ""))
    db_uid = st.text_input("UID (User ID)", value=os.getenv("UID", ""))
    db_pwd = st.text_input("PWD (Password)", type="password", value=os.getenv("PWD", ""))
    
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
                    st.session_state.schema = fetch_schema(conn)
                    st.session_state.schema_text = schema_to_prompt_text(st.session_state.schema, db_database)
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

def render_assistant_elements(message):
    if "info" in message: 
        st.info(message["info"])
        return

    if "sql" in message and message["sql"]:
        with st.expander("📄 View Generated SQL Query", expanded=False):
            display_sql = message["sql"].replace("TOP 100000", "TOP 100").replace("top 100000", "TOP 100")
            st.code(display_sql, language="sql")
            
    if "df" in message and message["df"] is not None:
        df = message["df"]
        st.markdown("***💡 Data Results (Showing Top 100 Rows)***")
        st.dataframe(df.head(100), use_container_width=True)
        
        if message.get("show_ts") and message.get("has_ts"):
            try:
                df_ts = df.copy()
                date_col = message["date_col"]
                val_col = message["val_col"]
                
                if df_ts[date_col].dtype in ['int64', 'float64']:
                    df_ts[date_col] = df_ts[date_col].astype(str)
                    
                df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
                df_ts = df_ts.dropna(subset=[date_col]).sort_values(by=date_col)
                
                fig_ts = px.line(df_ts, x=date_col, y=val_col, title="📈 Time Series Trend & Moving Average", markers=True)
                fig_ts.add_scatter(x=df_ts[date_col], y=df_ts[val_col].rolling(window=2, min_periods=1).mean(), name="Moving Avg")
                st.plotly_chart(fig_ts, use_container_width=True)
            except: pass

        if message.get("show_chart") and message.get("chart_code"):
            try:
                df_chart = df.copy()
                for col in df_chart.columns:
                    if 'year' in col.lower() or 'id' in col.lower() or 'calendar' in col.lower():
                        df_chart[col] = df_chart[col].astype(str)
                        
                ldict = {"df": df_chart, "px": px}
                exec(message["chart_code"], {}, ldict)
                st.plotly_chart(ldict["fig"], use_container_width=True)
            except: pass
            
    if message.get("show_analysis") and message.get("analysis"):
        st.markdown("***🚀 Strategic Insights (Full Dataset Analysis)***")
        st.markdown(message["analysis"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message: 
            st.write(message["content"])
        if message["role"] == "assistant":
            render_assistant_elements(message)


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
                "role": "assistant", "content": f"Processed query: {user_query}", "sql": "", "df": None, "analysis": "", "chart_code": "", 
                "has_ts": False, "show_chart": False, "show_ts": False, "show_analysis": False
            }
            
            try:
                language_instruction = (
                    "Strict Rule: Always analyze the user's input language from the query and continuous conversation history. "
                    "If they write in Urdu script or Roman Urdu (English text representing Urdu words like 'karo', 'dikhao', 'mujhe'), "
                    "you MUST write the final strategic insights, summary text, chart titles, and bullet points in Urdu / Roman Urdu format. "
                    "Keep pure technical jargon and dimensional column headers in English (e.g., Orders, Customers, Growth, Trend) "
                    "but write all structural prose and connecting explanations in Urdu / Roman Urdu so it perfectly aligns with their input style. "
                    "If the query is purely in English, write the response entirely in English."
                )
                
                with st.status("🧠 thinking...", expanded=False) as status:
                    decision = intelligent_agent_decision(user_query, st.session_state.schema_text, current_history, groq_api_key)
                    
                    if decision.get("out_of_scope"):
                        status.update(label="Out of Scope information not Provided in the Schema / Table", state="error")
                        err_text = "Out of scope: Information not found in the provided tables / schema."
                        msg_data["info"] = err_text
                    else:
                        msg_data["sql"] = decision.get("sql", "")
                        
                        tier = decision.get("recommended_model_tier", "fast")
                        active_model = MODELS_POOL.get(tier, MODELS_POOL["default"])
                        
                        conn = get_connection(db_driver, db_server, db_database, db_uid, db_pwd)
                        cursor = conn.cursor()
                        cursor.execute(msg_data["sql"])
                        data = rows_to_dicts(cursor)
                        
                        if data:
                            df = pd.DataFrame(data)
                            msg_data["df"] = df
                            
                            has_enough_data = len(df) > 1 and len(df.columns) > 1
                            
                            msg_data["show_chart"] = decision.get("needs_chart", False) or has_enough_data
                            msg_data["show_ts"] = decision.get("needs_timeseries", False)
                            msg_data["show_analysis"] = decision.get("needs_analysis", False) or has_enough_data
                            
                            if msg_data["show_ts"]:
                                date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower() or 'month' in c.lower()]
                                numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                                if date_cols and numeric_cols:
                                    msg_data["has_ts"] = True
                                    msg_data["date_col"] = date_cols[0]
                                    msg_data["val_col"] = numeric_cols[0]

                            if msg_data["show_chart"]:
                                chart_code = generate_chart_logic(df, groq_api_key, active_model, language_instruction)
                                msg_data["chart_code"] = chart_code

                            if msg_data["show_analysis"]:
                                numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                                if len(df) > 200:
                                    summary_str = f"Total Records Found: {len(df)}\n" + df.describe().to_markdown() + "\n\n"
                                    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
                                    if cat_cols and numeric_cols:
                                        summary_str += df.groupby(cat_cols[0])[numeric_cols[0]].sum().nlargest(15).to_markdown()
                                    analysis = generate_advanced_analysis_from_summary(summary_str, groq_api_key, active_model, language_instruction, current_history)
                                else:
                                    analysis = generate_advanced_analysis(df, groq_api_key, active_model, language_instruction, current_history)
                                msg_data["analysis"] = analysis

                render_assistant_elements(msg_data)
                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Error occurred: {e}")