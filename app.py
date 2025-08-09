import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Journal Insight Dashboard",
    page_icon="📔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Point to your pipeline's output file
    return pd.read_csv("data/processed/cleaned_journal_features.csv")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Journal Filter")
periods = df['date'].unique()
selected_period = st.sidebar.selectbox("Select Period/Month", options=periods)

traits = [col for col in df.columns if col.startswith('big5_')]
cluster_col = "quirk_cluster" if "quirk_cluster" in df.columns else None
peer_col = "peer_group" if "peer_group" in df.columns else None

# --- FILTER DATA ---
df_period = df[df['date'] == selected_period]

# --- TITLE AND INFO ---
st.title("📔 Journal Analysis Dashboard")
st.markdown("Explore your journaling data: emotions, cognitive patterns, traits, and more.")

# --- METRICS: Top Emotions, Traits, Clusters ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Top Emotion", df_period['emotion'].mode().iloc[0] if not df_period.empty else "N/A")
with col2:
    st.metric("Peer Group", df_period[peer_col].mode().iloc[0] if peer_col and not df_period.empty else "N/A")
with col3:
    st.metric("Quirk Cluster", df_period[cluster_col].mode().iloc[0] if cluster_col and not df_period.empty else "N/A")

# --- SHOW JOURNALS TABLE ---
st.subheader("Journal Entries - Current Period")
st.dataframe(df_period[['date', 'text', 'emotion', 'bias/distortion', 'feedback/insight', 'quirk_cluster', 'peer_group'] + traits].head(20))

# --- EMOTION/PATTERN TRENDS ---
st.subheader("Emotion Trend Over Time")
fig1 = px.line(df, x="date", y="emotion", title="Emotion Change Over Time", markers=True)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Distortion Count Trend Over Time")
fig2 = px.line(df, x="date", y="distortion_count", title="Distortion Frequency Over Time")
st.plotly_chart(fig2, use_container_width=True)

# --- TRAIT VISUALIZATION ---
st.subheader("Big Five Traits by Period")
if traits:
    df_trait = df.groupby('date')[traits].mean().reset_index()
    fig3 = px.line(df_trait, x="date", y=traits, title="Trait Trends")
    st.plotly_chart(fig3, use_container_width=True)

# --- FEEDBACK SECTION ---
st.subheader("Automated Feedback / Insights")
for idx, row in df_period.iterrows():
    st.info(f"**{row['date']} — Emotion:** {row['emotion']}\n\n**Feedback:** {row['feedback/insight']}")

# --- DOWNLOAD/EXPORT ---
st.sidebar.markdown("---")
st.sidebar.header("Export")
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download Full Data", csv, "journal_features.csv", "text/csv")

st.sidebar.markdown("*Made with Streamlit — Journal Analysis Model Dashboard*")
