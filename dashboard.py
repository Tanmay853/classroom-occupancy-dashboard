import streamlit as st
import pandas as pd
import plotly.express as px
import time
from supabase import create_client
import cv2
# ================= CONFIG =================
# SUPABASE_URL = "https://zsbieljcnrndkynpbtjf.supabase.co" - EXPIRED
SUPABASE_URL = "https://npttlzmbenjoydbhuadk.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpzYmllbGpjbnJuZGt5bnBidGpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg2OTY1OTEsImV4cCI6MjA3NDI3MjU5MX0.eAGEyXcIjwWFS-t_P5u1WMDMdkgEwhJ2S1juWEq6wuo" - EXPIRED
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wdHRsem1iZW5qb3lkYmh1YWRrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAzNzYwNzMsImV4cCI6MjA4NTk1MjA3M30.wq2qtAfD7oVyftqESPcnBEooVAOw1F4OqgTaf3FPsWQ"

import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

REFRESH_SEC = 10
LAYOUT_IMAGE = "lc001_borders.png"              # floor plan image

# ================= SETUP =================
st.set_page_config(
    page_title="Classroom Occupancy Dashboard",
    layout="wide"
)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= DATA FETCH =================
@st.cache_data(ttl=REFRESH_SEC)
def load_data():
    res = (
        supabase
        .table("occupancy")
        .select("*")
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    return pd.DataFrame(res.data)

df = load_data()

if df.empty:
    st.warning("No data found in database.")
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("Controls")
rooms = df["room"].unique()
selected_room = st.sidebar.selectbox("Select Room", rooms)

df_room = df[df["room"] == selected_room].sort_values("created_at", ascending=False)
latest = df_room.iloc[0]

# ================= HEADER =================
st.title("📊 Classroom Occupancy Dashboard")
st.caption(f"Room: **{selected_room}** | Last updated: {latest['created_at']}")

# ================= METRICS =================
col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Students", latest["total_count"])
col2.metric("📍 Zones Active", sum(latest[f"zone{i+1}"] > 0 for i in range(6)))
col3.metric("⏱️ Auto Refresh", f"{REFRESH_SEC}s")

# ================= ZONE BAR CHART =================
zone_values = {
    f"Zone {i+1}": latest[f"zone{i+1}"] for i in range(6)
}

fig_bar = px.bar(
    x=list(zone_values.keys()),
    y=list(zone_values.values()),
    labels={"x": "Zone", "y": "Students"},
    title="Zone-wise Occupancy"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ================= FLOOR PLAN =================
st.subheader("🗺️ Floor Plan View")
img = cv2.imread(LAYOUT_IMAGE)
img = cv2.resize(img, (600, 600))
st.image(img, channels="BGR")

zone_cols = st.columns(6)
for i, col in enumerate(zone_cols):
    col.metric(f"Zone {i+1}", latest[f"zone{i+1}"])

# ================= TIME SERIES =================
st.subheader("📈 Occupancy Over Time")

fig_line = px.line(
    df_room.sort_values("created_at"),
    x="created_at",
    y="total_count",
    title="Total Occupancy Trend"
)

st.plotly_chart(fig_line, use_container_width=True)

# ================= AUTO REFRESH =================
time.sleep(REFRESH_SEC)
st.rerun()
