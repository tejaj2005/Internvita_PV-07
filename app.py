# app_v2.py - ENHANCED DATA PREVIEW
# PowerGrid Analytics - Production Grade with Full-Screen Data View

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import uuid
import openpyxl
from db_manager import (
    init_db,
    register_user,
    authenticate_user,
    get_user_by_id,
    save_session_data,
    load_session_data,
)


# =============================================================================
# CONFIG & SETUP
# =============================================================================


API_BASE = "http://localhost:8000"
init_db()


st.set_page_config(
    page_title="⚡ PowerGrid Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS - CLEAN MODERN DESIGN
# =============================================================================


st.markdown(
    """
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header */
.header-container {
    background: linear-gradient(135deg, #dc2626 0%, #0369a1 100%);
    padding: 2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    color: white;
}

.header-container h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.header-container p {
    font-size: 1.1rem;
    opacity: 0.95;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #334155;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: #dc2626;
    box-shadow: 0 8px 20px rgba(220, 38, 38, 0.2);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #dc2626;
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.9rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Login/Signup Form */
.auth-container {
    max-width: 420px;
    margin: 80px auto;
    padding: 2rem;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-radius: 12px;
    border: 1px solid #334155;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
}

.auth-header {
    text-align: center;
    margin-bottom: 2rem;
}

.auth-header h2 {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #dc2626, #0369a1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.auth-header p {
    color: #94a3b8;
    font-size: 0.95rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    padding: 12px 24px;
    background: #1e293b;
    border-radius: 8px;
    border: 1px solid #334155;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #dc2626, #0369a1);
    border: none;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #dc2626 0%, #0369a1 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(220, 38, 38, 0.4);
}

/* Input Fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stNumberInput > div > div > input {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stNumberInput > div > div > input:focus {
    border-color: #dc2626 !important;
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.1) !important;
}

/* Success/Error Messages */
.stSuccess {
    background: #064e3b !important;
    border: 1px solid #10b981 !important;
    color: #a7f3d0 !important;
    padding: 12px !important;
    border-radius: 8px !important;
}

.stError {
    background: #7f1d1d !important;
    border: 1px solid #ef4444 !important;
    color: #fca5a5 !important;
    padding: 12px !important;
    border-radius: 8px !important;
}

.stInfo {
    background: #0c2d6b !important;
    border: 1px solid #3b82f6 !important;
    color: #bfdbfe !important;
    padding: 12px !important;
    border-radius: 8px !important;
}

/* Sidebar */
.stSidebar {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

/* Charts */
.plotly-graph-div {
    border-radius: 10px;
    border: 1px solid #334155;
    overflow: hidden;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(220, 38, 38, 0.1) !important;
    border: 1px solid #dc2626 !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(220, 38, 38, 0.2) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION INITIALIZATION WITH URL PERSISTENCE
# =============================================================================


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "session_token" not in st.session_state:
    st.session_state.session_token = str(uuid.uuid4())
if "df_uploaded" not in st.session_state:
    st.session_state.df_uploaded = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False


# =============================================================================
# RESTORE LOGIN FROM URL PARAMS (survives F5/Cmd+R)
# Compatible with Streamlit 1.10+
# =============================================================================


def get_query_params():
    """Get query params - compatible with all Streamlit versions"""
    try:
        return dict(st.query_params)
    except AttributeError:
        try:
            return st.experimental_get_query_params()
        except AttributeError:
            return {}


def set_query_params(params):
    """Set query params - compatible with all Streamlit versions"""
    try:
        st.query_params.update(params)
    except AttributeError:
        try:
            st.experimental_set_query_params(**params)
        except AttributeError:
            pass


def restore_from_url_params():
    """Check if user_id in URL and restore authentication"""
    params = get_query_params()
    
    if "user_id" in params and "username" in params:
        user_id = params["user_id"]
        if isinstance(user_id, list):
            user_id = user_id[0]
        
        username = params["username"]
        if isinstance(username, list):
            username = username[0]
        
        st.session_state.authenticated = True
        st.session_state.user_id = user_id
        st.session_state.username = username
        
        data = load_session_data(user_id)
        if data:
            st.session_state.df_uploaded = data.get("df_uploaded")
            st.session_state.df_processed = data.get("df_processed")
            st.session_state.predictions = data.get("predictions")


restore_from_url_params()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


REQUIRED_COLUMNS_DEFAULTS = {
    "hour": 0,
    "day_of_week": 0,
    "month": 1,
    "electricity_usage_kWh": 0.0,
    "temp": 25.0,
    "humidity": 60.0,
}


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        if "hour" not in df.columns:
            df["hour"] = dt.dt.hour.fillna(0).astype(int)
        if "day_of_week" not in df.columns:
            df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
        if "month" not in df.columns:
            df["month"] = dt.dt.month.fillna(1).astype(int)

    for col, default in REQUIRED_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    if "consumption" not in df.columns:
        if "electricity_usage_kWh" in df.columns:
            df["consumption"] = df["electricity_usage_kWh"]
        else:
            df["consumption"] = 5000.0

    if "temperature" not in df.columns:
        if "temp" in df.columns:
            df["temperature"] = df["temp"]
        else:
            df["temperature"] = 25.0

    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown", inplace=True)
    return df


def preprocess_numeric(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    df_proc = df.copy()
    num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["electricity_usage_kWh", "actual", "forecast", "consumption"]:
        if col in num_cols:
            num_cols.remove(col)
    if num_cols:
        scaler = StandardScaler()
        df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])
    return df_proc


def save_user_session():
    """Save session to database for persistence across refreshes"""
    if st.session_state.user_id:
        save_session_data(
            st.session_state.user_id,
            st.session_state.session_token,
            {
                "df_uploaded": st.session_state.df_uploaded,
                "df_processed": st.session_state.df_processed,
                "predictions": st.session_state.predictions,
            },
        )


# =============================================================================
# AUTH PAGES
# =============================================================================


def login_signup_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
<div class="auth-container">
  <div class="auth-header">
    <h2>⚡ PowerGrid</h2>
    <p>Energy Load Forecasting & Anomaly Detection</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if not st.session_state.show_signup:
            st.markdown("### 🔐 Login")
            username = st.text_input("👤 Username", key="login_username")
            password = st.text_input("🔑 Password", type="password", key="login_password")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Login", use_container_width=True):
                    if not username or not password:
                        st.error("⚠️ Enter username and password")
                    else:
                        success, user_id = authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user_id = user_id
                            st.session_state.username = username
                            
                            set_query_params({"user_id": user_id, "username": username})
                            
                            restore_from_url_params()
                            st.success("✅ Logged in successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
            with col_b:
                if st.button("Create Account", use_container_width=True):
                    st.session_state.show_signup = True
                    st.rerun()

        else:
            st.markdown("### 📝 Create Account")
            new_username = st.text_input("👤 Username", key="signup_username")
            new_email = st.text_input("📧 Email", key="signup_email")
            new_password = st.text_input("🔑 Password", type="password", key="signup_password")
            confirm_password = st.text_input(
                "🔑 Confirm Password", type="password", key="confirm_password"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Sign Up", use_container_width=True):
                    if not new_username or not new_email or not new_password:
                        st.error("⚠️ Fill all fields")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords don't match")
                    else:
                        success, msg = register_user(new_username, new_email, new_password)
                        if success:
                            st.success(msg)
                            st.session_state.show_signup = False
                            st.rerun()
                        else:
                            st.error(msg)
            with col_b:
                if st.button("Back to Login", use_container_width=True):
                    st.session_state.show_signup = False
                    st.rerun()


# =============================================================================
# MAIN APP
# =============================================================================


if not st.session_state.authenticated:
    login_signup_page()
    st.stop()


st.markdown(
    """
<div class="header-container">
  <h1>⚡ PowerGrid Analytics</h1>
  <p>Smart Energy Load Forecasting & Real-time Anomaly Detection</p>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(f"### 👤 Welcome, **{st.session_state.username}**!")

    if st.button("🚪 Logout", use_container_width=True):
        save_user_session()
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.df_uploaded = None
        st.session_state.df_processed = None
        st.session_state.predictions = None
        
        set_query_params({})
        st.rerun()

    st.divider()

    if st.session_state.df_uploaded is not None:
        st.divider()
        st.markdown("### 📈 Dataset Info")
        st.metric("Records", f"{len(st.session_state.df_uploaded):,}")
        st.metric("Columns", st.session_state.df_uploaded.shape[1])
        st.metric("Memory", f"{st.session_state.df_uploaded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.df_uploaded = None
            st.session_state.df_processed = None
            st.session_state.predictions = None
            save_user_session()
            st.rerun()


tabs = st.tabs(
    ["📤 Data Upload", "📊 Analytics", "🚨 Anomalies", "🎯 Forecast", "📥 Reports"]
)


# =============================================================================
# TAB 1: DATA UPLOAD (ENHANCED)
# =============================================================================


with tabs[0]:
    st.markdown("## 📤 Data Management")

    st.markdown("### 📤 Upload CSV File")
    st.info(
        "Upload your smart meter data. Supported formats: CSV files with "
        "electricity consumption, date/time, and optional weather data."
    )

    uploaded = st.file_uploader(
        "Choose CSV file", type="csv", help="Max 200MB per file"
    )

    if uploaded is not None:
        with st.spinner("Processing..."):
            prog = st.progress(0, text="🔄 Reading file...")
            try:
                try:
                    df = pd.read_csv(uploaded, on_bad_lines="skip")
                except TypeError:
                    df = pd.read_csv(uploaded)

                prog.progress(25, text="📋 Validating structure...")
                df = ensure_required_columns(df)

                prog.progress(50, text="🧹 Imputing missing values...")
                df = impute_missing_values(df)

                prog.progress(75, text="⚙️ Normalizing features...")
                df_proc = preprocess_numeric(df)

                prog.progress(100, text="✅ Complete!")
                st.session_state.df_uploaded = df
                st.session_state.df_processed = df_proc
                save_user_session()
                st.success("✅ Data uploaded and preprocessed!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                prog.empty()

    if st.session_state.df_uploaded is not None:
        st.divider()
        st.markdown("### 📋 Data Preview")
        
        # Create two columns: Preview + Quick Stats
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Expandable full data view
            with st.expander("📊 **View All Data** (Click to expand full dataset)", expanded=False):
                st.markdown("**Complete Dataset - Scroll to see all rows and columns**")
                st.dataframe(
                    st.session_state.df_uploaded,
                    use_container_width=True,
                    height=600
                )
            
            # Default first 10 rows
            st.markdown("**First 10 Rows Preview**")
            st.dataframe(
                st.session_state.df_uploaded.head(100000),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.markdown("**📈 Dataset Stats**")
            st.metric("Total Rows", f"{len(st.session_state.df_uploaded):,}")
            st.metric("Total Cols", st.session_state.df_uploaded.shape[1])
            st.metric("Memory Usage", f"{st.session_state.df_uploaded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.markdown("**📝 Column Names**")
            for i, col in enumerate(st.session_state.df_uploaded.columns):
                if i < 15:
                    st.text(f"• {col}")
            if len(st.session_state.df_uploaded.columns) > 15:
                st.text(f"... + {len(st.session_state.df_uploaded.columns) - 15} more")
        
        st.divider()
        
        # Download section
        st.markdown("### 📥 Download Data")
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            csv_data = st.session_state.df_uploaded.to_csv(index=False)
            st.download_button(
                "📥 CSV",
                data=csv_data,
                file_name=f"powergrid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col_d2:
            # Generate Excel file
            from io import BytesIO
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                st.session_state.df_uploaded.to_excel(writer, index=False)
            excel_buffer.seek(0)
            st.download_button(
                "📥 Excel",
                data=excel_buffer.getvalue(),
                file_name=f"powergrid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        
        with col_d3:
            json_data = st.session_state.df_uploaded.to_json(orient="records", indent=2)
            st.download_button(
                "📥 JSON",
                data=json_data,
                file_name=f"powergrid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )


# =============================================================================
# TAB 2: ANALYTICS
# =============================================================================


with tabs[1]:
    st.markdown("## 📊 Load Analytics & Insights")

    df = st.session_state.df_uploaded
    if df is None:
        st.info("👉 Upload data in the **Data Upload** tab first")
    else:
        st.markdown("### 🎯 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)

        if "consumption" in df.columns:
            cons = df["consumption"].dropna()
            m1.metric("Average Load", f"{cons.mean():.0f} kWh", "Daily average")
            m2.metric("Peak Load", f"{cons.max():.0f} kWh", "Maximum recorded")
            m3.metric("Min Load", f"{cons.min():.0f} kWh", "Minimum recorded")
            m4.metric("Std Deviation", f"{cons.std():.0f} kWh", "Load variability")

        st.divider()

        st.markdown("### 📈 Load Patterns")
        col1, col2 = st.columns(2)

        if "hour" in df.columns and "consumption" in df.columns:
            with col1:
                st.markdown("**Hourly Load Pattern**")
                hourly = df.groupby("hour")["consumption"].mean().reset_index()
                fig_h = go.Figure()
                fig_h.add_trace(
                    go.Scatter(
                        x=hourly["hour"],
                        y=hourly["consumption"],
                        mode="lines+markers",
                        name="Average Load",
                        line=dict(color="#dc2626", width=3),
                        marker=dict(size=6),
                    )
                )
                fig_h.update_layout(
                    title="<b>Average Consumption by Hour</b><br><sub>Shows peak demand times (typically morning/evening)</sub>",
                    xaxis_title="Hour of Day",
                    yaxis_title="kWh",
                    template="plotly_dark",
                    hovermode="x unified",
                    height=400,
                )
                st.plotly_chart(fig_h, use_container_width=True)

        if "day_of_week" in df.columns and "consumption" in df.columns:
            with col2:
                st.markdown("**Daily Load Pattern**")
                daily = df.groupby("day_of_week")["consumption"].mean().reset_index()
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily["day_name"] = daily["day_of_week"].map(lambda x: days[x] if x < 7 else x)

                fig_d = go.Figure()
                fig_d.add_trace(
                    go.Bar(
                        x=daily["day_name"],
                        y=daily["consumption"],
                        marker_color="#0369a1",
                        name="Average Load",
                    )
                )
                fig_d.update_layout(
                    title="<b>Consumption by Day of Week</b><br><sub>Weekend vs weekday comparison</sub>",
                    xaxis_title="Day",
                    yaxis_title="kWh",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig_d, use_container_width=True)

        if "consumption" in df.columns:
            st.markdown("### 📊 Load Distribution")
            st.markdown(
                "This histogram shows how electricity consumption is distributed. "
                "A normal distribution suggests balanced load, while skewed patterns indicate specific usage behaviors."
            )
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Histogram(
                    x=df["consumption"],
                    nbinsx=40,
                    marker_color="#dc2626",
                    name="Consumption",
                )
            )
            fig_dist.update_layout(
                title="<b>Consumption Distribution</b>",
                xaxis_title="kWh",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# TAB 3: ANOMALIES
# =============================================================================


with tabs[2]:
    st.markdown("## 🚨 Anomaly Detection")

    df = st.session_state.df_uploaded
    if df is None or "consumption" not in df.columns:
        st.info("Need data with consumption column")
    else:
        st.markdown(
            "**Anomalies are identified using statistical methods (Z-score > 2.5).** "
            "These represent unusual consumption patterns that deviate significantly from normal behavior."
        )

        cons = df["consumption"].dropna()
        z = np.abs((cons - cons.mean()) / cons.std())
        mask = z > 2.5
        anomalies = cons[mask]

        m1, m2, m3 = st.columns(3)
        m1.metric("Anomalies Detected", int(mask.sum()), f"{100*mask.mean():.2f}% of data")
        m2.metric("Normal Records", int(len(cons) - mask.sum()), "Regular patterns")
        m3.metric("Anomaly Severity", f"{z.max():.2f}σ", "Highest deviation")

        st.divider()

        st.markdown("### 📍 Anomaly Timeline")
        st.markdown(
            "Red dots indicate unusual consumption values. "
            "Green dots represent normal consumption. Clusters of anomalies suggest systematic issues."
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.where(~mask)[0],
                y=cons[~mask],
                mode="markers",
                marker=dict(color="#10b981", size=4, opacity=0.6),
                name="Normal",
            )
        )
        if anomalies.any():
            fig.add_trace(
                go.Scatter(
                    x=np.where(mask)[0],
                    y=anomalies,
                    mode="markers",
                    marker=dict(color="#dc2626", size=8, symbol="star"),
                    name="Anomaly",
                )
            )
        fig.update_layout(
            title="<b>Anomaly Detection Results</b>",
            xaxis_title="Record Index",
            yaxis_title="Consumption (kWh)",
            template="plotly_dark",
            hovermode="closest",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        if anomalies.any():
            st.markdown("### 🔍 Top Anomalies")
            df_an = pd.DataFrame(
                {"Index": anomalies.index, "Consumption (kWh)": anomalies.values}
            ).sort_values("Consumption (kWh)", ascending=False).head(20)
            st.dataframe(df_an, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: FORECAST
# =============================================================================

with tabs[3]:
    st.markdown("## 🎯 Load Forecasting")

    df = st.session_state.df_uploaded
    if df is None:
        st.info("👉 Upload data in the **Data Upload** tab first")
    else:
        st.markdown(
            "**ML-based forecasting** predicts future electricity demand. "
            "When you open this tab, the app automatically generates 24h, 7d and "
            "30d forecasts once and shows them. You can also request a custom "
            "horizon from 1 to 365 days."
        )

        col1, col2 = st.columns(2)
        with col1:
            primary_horizon = st.selectbox(
                "Primary Horizon to Display",
                ["24 Hours", "7 Days", "30 Days"],
                help="This horizon is used for the main chart and metrics.",
            )
        with col2:
            conf = st.slider(
                "Confidence Level",
                0.80,
                0.99,
                0.95,
                0.01,
                help="Higher = more conservative predictions",
            )

        # ---------- helper to call backend ----------
        def _call_backend(h_label: str):
            csv_buf = df.to_csv(index=False)
            resp = requests.post(
                f"{API_BASE}/forecast",
                json={"data": csv_buf, "horizon": h_label, "confidence": conf},
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"API Error ({h_label}): {resp.text}")
            data = resp.json()
            return data.get("predictions", [])

        # ---------- AUTO-GENERATE 24h / 7d / 30d ON FIRST VISIT ----------
        if "forecast_24h" not in st.session_state or "forecast_7d" not in st.session_state or "forecast_30d" not in st.session_state:
            with st.spinner("Auto-generating 24h, 7d and 30d forecasts..."):
                try:
                    preds_24 = _call_backend("24 Hours")
                    preds_7d = _call_backend("7 Days")
                    preds_30d = _call_backend("30 Days")

                    st.session_state["forecast_24h"] = {"label": "24 Hours", "predictions": preds_24}
                    st.session_state["forecast_7d"] = {"label": "7 Days", "predictions": preds_7d}
                    st.session_state["forecast_30d"] = {"label": "30 Days", "predictions": preds_30d}

                    # default report mode = builtin (24/7/30)
                    st.session_state["report_mode"] = "builtin"

                    # also store one horizon for quick display
                    st.session_state["predictions"] = preds_24
                    st.session_state["selected_horizon"] = "24 Hours"
                except Exception as e:
                    st.error(f"❌ Auto-forecast error: {e}")

        # ---------- show chosen primary horizon from already stored forecasts ----------
        f24 = st.session_state.get("forecast_24h")
        f7 = st.session_state.get("forecast_7d")
        f30 = st.session_state.get("forecast_30d")

        preds_show = []
        if primary_horizon == "24 Hours" and f24:
            preds_show = f24["predictions"]
        elif primary_horizon == "7 Days" and f7:
            preds_show = f7["predictions"]
        elif primary_horizon == "30 Days" and f30:
            preds_show = f30["predictions"]

        if preds_show:
            st.success(f"✅ Auto-forecast ready for {primary_horizon} ({len(preds_show)} points)")

            st.session_state["predictions"] = preds_show
            st.session_state["selected_horizon"] = primary_horizon

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average Forecast", f"{np.mean(preds_show):.0f} kWh")
            m2.metric("Peak Forecast", f"{np.max(preds_show):.0f} kWh")
            m3.metric("Min Forecast", f"{np.min(preds_show):.0f} kWh")
            m4.metric("Total Energy", f"{np.sum(preds_show):.0f} kWh")

            st.divider()
            st.markdown("### 📈 Forecast Curve")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(preds_show))),
                    y=preds_show,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#dc2626", width=3),
                    marker=dict(size=5),
                    fill="tozeroy",
                    fillcolor="rgba(220, 38, 38, 0.1)",
                    hovertemplate="<b>Hour %{x}</b><br>Load: %{y:.0f} kWh<extra></extra>",
                )
            )
            fig.update_layout(
                title=(
                    f"<b>Load Forecast ({primary_horizon})</b>"
                    f"<br><sub>{len(preds_show)} predictions generated</sub>"
                ),
                xaxis_title="Time Period (hours)",
                yaxis_title="kWh",
                template="plotly_dark",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------- OPTIONAL CUSTOM HORIZON (separate button) ----------
        st.markdown("### 🔧 Optional Custom Horizon")

        custom_days = st.number_input(
            "Extra forecast horizon (days, optional)",
            min_value=1,
            max_value=365,
            value=7,
            step=1,
            help="Set an extra horizon between 1 and 365 days.",
        )

        st.info(
            "📌 Custom horizon = days × 24 hourly predictions. "
            "If you generate a custom forecast, the report will use ONLY that horizon."
        )

        if st.button("📆 Generate Custom Forecast", use_container_width=True):
            with st.spinner(f"Computing forecast for {int(custom_days)} days..."):
                try:
                    label = f"{int(custom_days)} Days"
                    preds_custom = _call_backend(label)

                    st.session_state["forecast_custom"] = {
                        "label": label,
                        "days": int(custom_days),
                        "predictions": preds_custom,
                    }
                    # switch report mode to custom
                    st.session_state["report_mode"] = "custom"

                    st.success(
                        f"✅ Generated {len(preds_custom)} predictions for {int(custom_days)}-day horizon"
                    )

                    # (optional) quick stats display
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg Forecast", f"{np.mean(preds_custom):.0f} kWh")
                    c2.metric("Peak", f"{np.max(preds_custom):.0f} kWh")
                    c3.metric("Min", f"{np.min(preds_custom):.0f} kWh")
                    c4.metric("Total", f"{np.sum(preds_custom):.0f} kWh")
                except Exception as e:
                    st.error(f"❌ Custom forecast error: {e}")



# =============================================================================
# TAB 5: REPORTS
# =============================================================================

with tabs[4]:
    st.markdown("## 📥 Reports & Export")

    df = st.session_state.df_uploaded

    # choose preprocessed df if available
    df_proc = None
    if "df_processed_cleaned" in st.session_state and st.session_state["df_processed_cleaned"] is not None:
        df_proc = st.session_state["df_processed_cleaned"]
    elif "df_processed" in st.session_state and st.session_state["df_processed"] is not None:
        df_proc = st.session_state["df_processed"]

    if df is not None and "consumption" in df.columns:
        st.markdown("### 📊 Summary Report")

        cons = df["consumption"].dropna()
        z = np.abs((cons - cons.mean()) / cons.std())
        anomaly_count = int((z > 2.5).sum())
        anomaly_rate = 100 * (z > 2.5).mean()

        # forecasts from session (dicts with 'predictions' list)
        f24 = st.session_state.get("forecast_24h")
        f7 = st.session_state.get("forecast_7d")
        f30 = st.session_state.get("forecast_30d")
        fcustom = st.session_state.get("forecast_custom")

        # convert to numpy for stats
        p24 = np.array(f24["predictions"]) if isinstance(f24, dict) and f24.get("predictions") else None
        p7 = np.array(f7["predictions"]) if isinstance(f7, dict) and f7.get("predictions") else None
        p30 = np.array(f30["predictions"]) if isinstance(f30, dict) and f30.get("predictions") else None
        pcust = np.array(fcustom["predictions"]) if isinstance(fcustom, dict) and fcustom.get("predictions") else None
        custom_days = fcustom.get("days") if isinstance(fcustom, dict) else None

        # ---- build report text (with Gemini, hard-coded key) ----
        def build_report_text():
            lines = [
                "# PowerGrid Analytics Report",
                f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"\n**User:** {st.session_state.username}",
                "\n## Dataset Summary",
                f"- Total Records: {len(df):,}",
                f"- Total Columns: {df.shape[1]}",
                f"- Date Range: {df.get('datetime', pd.Series()).min() if 'datetime' in df.columns else 'N/A'} "
                f"to {df.get('datetime', pd.Series()).max() if 'datetime' in df.columns else 'N/A'}",
                "\n## Load Statistics",
                f"- Mean Consumption: {cons.mean():.2f} kWh",
                f"- Median Consumption: {cons.median():.2f} kWh",
                f"- Peak Consumption: {cons.max():.2f} kWh",
                f"- Minimum Consumption: {cons.min():.2f} kWh",
                f"- Std Deviation: {cons.std():.2f} kWh",
                f"- Total Energy: {cons.sum():.2f} kWh",
                "\n## Anomaly Analysis",
                f"- Anomalies Detected: {anomaly_count}",
                f"- Anomaly Rate: {anomaly_rate:.2f}%",
            ]

            # ==================================================================
            # FORECAST SUMMARY LOGIC
            # ==================================================================
            report_mode = st.session_state.get("report_mode", "builtin")

            # choose which arrays to show
            p24_use = p7_use = p30_use = pcust_use = None
            custom_days_use = custom_days

            if report_mode == "custom" and pcust is not None:
                # user used custom mode → ONLY custom block in report
                lines.append("\n## Forecast Summary")
                pcust_use = pcust
            else:
                # default → show 24h/7d/30d if available
                if any(v is not None for v in [p24, p7, p30]):
                    lines.append("\n## Forecast Summary")
                p24_use = p24
                p7_use = p7
                p30_use = p30

            if p24_use is not None:
                lines.extend([
                    "\n### 24-Hour Forecast",
                    f"- Average Forecast: {p24_use.mean():.2f} kWh",
                    f"- Peak Forecast: {p24_use.max():.2f} kWh",
                    f"- Total Energy (24h): {p24_use.sum():.2f} kWh",
                ])

            if p7_use is not None:
                lines.extend([
                    "\n### 7-Day Forecast",
                    f"- Average Forecast: {p7_use.mean():.2f} kWh",
                    f"- Peak Forecast: {p7_use.max():.2f} kWh",
                    f"- Total Energy (7d): {p7_use.sum():.2f} kWh",
                ])

            if p30_use is not None:
                lines.extend([
                    "\n### 30-Day Forecast",
                    f"- Average Forecast: {p30_use.mean():.2f} kWh",
                    f"- Peak Forecast: {p30_use.max():.2f} kWh",
                    f"- Total Energy (30d): {p30_use.sum():.2f} kWh",
                ])

            if pcust_use is not None and custom_days_use is not None:
                lines.extend([
                    f"\n### {custom_days_use}-Day Custom Forecast",
                    f"- Average Forecast: {pcust_use.mean():.2f} kWh",
                    f"- Peak Forecast: {pcust_use.max():.2f} kWh",
                    f"- Total Energy ({custom_days_use}d): {pcust_use.sum():.2f} kWh",
                ])

            # ==================================================================
            # GEMINI AI SUMMARY – use the same chosen arrays
            # ==================================================================
            ai_text = None
            try:
                import google.generativeai as genai

                GEMINI_API_KEY = "AIzaSyDV-Nm3usHhpBcp9zx3CpvZwZgtDQ_JCcg"  # demo only

                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-2.5-flash")

                # safe means for whatever is active
                mean_24 = float(p24_use.mean()) if p24_use is not None and len(p24_use) > 0 else None
                mean_7  = float(p7_use.mean())  if p7_use  is not None and len(p7_use)  > 0 else None
                mean_30 = float(p30_use.mean()) if p30_use is not None and len(p30_use) > 0 else None
                mean_c  = float(pcust_use.mean()) if pcust_use is not None and len(pcust_use) > 0 else None

                mean_24_str = f"{mean_24:.2f}" if mean_24 is not None else "N/A"
                mean_7_str  = f"{mean_7:.2f}"  if mean_7  is not None else "N/A"
                mean_30_str = f"{mean_30:.2f}" if mean_30 is not None else "N/A"
                mean_c_str  = f"{mean_c:.2f}"  if mean_c  is not None else "N/A"

                prompt = (
                    "You are analysing an electricity demand forecasting report.\n\n"
                    f"Historical mean: {cons.mean():.2f} kWh\n"
                    f"Historical peak: {cons.max():.2f} kWh\n"
                    f"Anomalies: {anomaly_count} ({anomaly_rate:.2f}%)\n\n"
                    f"24h avg: {mean_24_str} kWh\n"
                    f"7d avg: {mean_7_str} kWh\n"
                    f"30d avg: {mean_30_str} kWh\n"
                    f"Custom avg: {mean_c_str} kWh\n\n"
                    "Give:\n"
                    "1) 2–3 sentence insight about demand pattern,\n"
                    "2) mention peaks / anomalies,\n"
                    "3) 1–2 actionable recommendations for grid operators.\n"
                    "Max 120 words, plain text."
                )

                resp = model.generate_content(prompt)
                ai_text = resp.text.strip()
            except Exception as e:
                st.error(f"Gemini error: {e}")
                ai_text = None

            if ai_text:
                lines.extend([
                    "\n## AI Summary (Gemini)",
                    ai_text,
                ])
            else:
                stability = "stable" if (cons.std() / cons.mean() < 0.15) else "variable"
                lines.extend([
                    "\n## AI Summary (Fallback)",
                    f"- Load appears {stability} (CV={cons.std()/cons.mean()*100:.1f}%).",
                    f"- Peak-to-average ratio: {cons.max()/cons.mean():.2f}x.",
                    f"- {anomaly_count} anomalies ({anomaly_rate:.2f}%) suggest "
                    f"{'good' if anomaly_rate < 1 else 'some'} data quality.",
                ])

            return "\n".join(lines)

        # build once so both preview and download use same text
        report_text = build_report_text()

        col1, col2 = st.columns(2)

        # single button: generates and downloads directly
        with col1:
            st.download_button(
                "📥 Download Report (TXT)",
                data=report_text.encode("utf-8"),
                file_name=f"powergrid_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # preprocessed data download (if available)
        with col2:
            target_df = df_proc if df_proc is not None else df
            csv_bytes = target_df.to_csv(index=False)
            st.download_button(
                "📥 Download Data (CSV)",
                data=csv_bytes,
                file_name=f"powergrid_preprocessed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # optional preview
        with st.expander("📄 Report Preview"):
            st.text_area("", report_text, height=350, disabled=True)

        st.divider()
        st.markdown("### 🔧 System Settings")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Session Manually", use_container_width=True):
                save_user_session()
                st.success("✅ Session saved to database")
        with c2:
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.df_uploaded = None
                st.session_state.df_processed = None
                st.session_state.predictions = None
                save_user_session()
                st.rerun()
    else:
        st.info("No data available. Upload data to create reports.")

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        """
**PowerGrid Analytics v2.0**

- 🔐 Secure authentication with SQLite database
- 💾 Automatic session persistence
- 📊 ML-based load forecasting (24h/7d/30d)
- 🚨 Real-time anomaly detection
- 📈 Professional data visualization

**Tech Stack:** Streamlit • FastAPI • SQLite • Plotly • Scikit-learn
"""
    )
