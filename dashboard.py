import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
import cv2
import os

# ================= CONFIG =================
REFRESH_SEC = 10
LAYOUT_IMAGE = "lc001_borders.png"

ZONE_CAPACITY = [20, 20, 40, 40, 20, 20]
OVERLOAD_THRESHOLD = 80  # %

# ================= SUPABASE (ENV ONLY) =================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Supabase credentials not set.")
    st.stop()

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
        .limit(300)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df

df = load_data()

if df.empty:
    st.warning("No data available in database.")
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("Controls")

rooms = df["room"].unique()
selected_room = st.sidebar.selectbox("Select Room", rooms)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 10 minutes", "Last 1 hour", "Today"]
)

df_room = df[df["room"] == selected_room].copy()
now = pd.Timestamp.now(tz="UTC")

if time_range == "Last 10 minutes":
    df_room = df_room[df_room["created_at"] >= now - pd.Timedelta(minutes=10)]
elif time_range == "Last 1 hour":
    df_room = df_room[df_room["created_at"] >= now - pd.Timedelta(hours=1)]
else:
    df_room = df_room[df_room["created_at"].dt.date == now.date()]

df_room = df_room.sort_values("created_at", ascending=False)

# ================= EMPTY WINDOW GUARD =================
if df_room.empty:
    st.info("No data in selected time range. Showing latest available data instead.")
    df_room = df[df["room"] == selected_room].sort_values(
        "created_at", ascending=False
    )

latest = df_room.iloc[0]
previous = df_room.iloc[1] if len(df_room) > 1 else latest

# ================= HEADER =================
st.title("📊 Classroom Occupancy Dashboard")
st.caption(f"Room: **{selected_room}** | Updated: {latest['created_at']}")

# ================= METRICS =================
c1, c2, c3 = st.columns(3)

c1.metric(
    "👥 Total Students",
    latest["total_count"],
    delta=latest["total_count"] - previous["total_count"]
)

active_zones = sum(latest[f"zone{i+1}"] > 0 for i in range(6))
c2.metric("📍 Active Zones", active_zones)

avg_util = sum(
    (latest[f"zone{i+1}"] / ZONE_CAPACITY[i]) * 100
    for i in range(6)
) / 6
c3.metric("📊 Avg Utilization", f"{avg_util:.1f}%")

# ================= UTILIZATION + ALERTS =================
st.subheader("⚠️ Zone Utilization & Alerts")

zone_names = [f"Zone {i+1}" for i in range(6)]
zone_occupied = [latest[f"zone{i+1}"] for i in range(6)]
zone_capacity = ZONE_CAPACITY

zone_util = [
    (zone_occupied[i] / zone_capacity[i]) * 100
    if zone_capacity[i] > 0 else 0
    for i in range(6)
]

zone_labels = [
    f"{zone_occupied[i]} / {zone_capacity[i]}"
    for i in range(6)
]

for i, util in enumerate(zone_util):
    if util > OVERLOAD_THRESHOLD:
        st.error(f"⚠️ Zone {i+1} overloaded ({util:.1f}%)")

# ================= UTILIZATION BAR (ENHANCED) =================
fig_bar = px.bar(
    x=zone_names,
    y=zone_util,
    text=zone_labels,
    labels={"x": "Zone", "y": "Utilization (%)"},
    title="Zone Utilization (%) with Occupied / Capacity",
    range_y=[0, 100]
)

fig_bar.update_traces(
    textposition="outside",
    hovertemplate=(
        "Zone: %{x}<br>"
        "Utilization: %{y:.1f}%<br>"
        "Occupied / Capacity: %{text}<extra></extra>"
    )
)

fig_bar.update_layout(
    uniformtext_minsize=10,
    uniformtext_mode="hide"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ================= FLOOR PLAN =================
st.subheader("🗺️ Floor Plan View")

img = cv2.imread(LAYOUT_IMAGE)
img = cv2.resize(img, (600, 600))
st.image(img, channels="BGR")

# ================= TIME SERIES =================
st.subheader("📈 Occupancy Over Time")

fig_line = px.line(
    df_room.sort_values("created_at"),
    x="created_at",
    y="total_count",
    title="Total Occupancy Trend"
)
st.plotly_chart(fig_line, use_container_width=True)

# ================= EXPLAINABILITY =================
with st.expander("ℹ️ How occupancy is computed"):
    st.markdown("""
    - Multi-camera person detection  
    - Homography projection to floor plan  
    - DBSCAN merging to avoid double counting  
    - Zone-wise counting  
    - Supabase-backed cloud dashboard  
    """)
