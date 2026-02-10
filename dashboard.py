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

# ================= SUPABASE =================
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
        .limit(500)
        .execute()
    )

    df = pd.DataFrame(res.data)
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["env_time"] = pd.to_datetime(df.get("env_time"), utc=True, errors="coerce")

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

# ================= FILTER =================
df_room = df[df["room"] == selected_room].copy()
now = pd.Timestamp.now(tz="UTC")

if time_range == "Last 10 minutes":
    df_room = df_room[df_room["created_at"] >= now - pd.Timedelta(minutes=10)]
elif time_range == "Last 1 hour":
    df_room = df_room[df_room["created_at"] >= now - pd.Timedelta(hours=1)]
else:
    df_room = df_room[df_room["created_at"].dt.date == now.date()]

df_room = df_room.sort_values("created_at", ascending=False)

# ================= FALLBACK =================
if df_room.empty:
    st.info("No data in selected range. Showing latest available data.")
    df_room = df[df["room"] == selected_room].sort_values("created_at", ascending=False)

latest = df_room.iloc[0]
previous = df_room.iloc[1] if len(df_room) > 1 else latest

# ================= ENV SYNC =================
env_df = df.dropna(subset=["env_time"]).sort_values("env_time")

def nearest_env(t):
    past = env_df[env_df["env_time"] <= t]
    return past.iloc[-1] if not past.empty else None

df_room["env_row"] = df_room["created_at"].apply(nearest_env)
df_room["temperature"] = df_room["env_row"].apply(
    lambda r: r["temperature"] if r is not None else None
)
df_room["humidity"] = df_room["env_row"].apply(
    lambda r: r["humidity"] if r is not None else None
)

latest_temp = df_room.iloc[0]["temperature"]
latest_hum = df_room.iloc[0]["humidity"]

# ================= HEADER =================
st.title("📊 Classroom Occupancy Dashboard")
st.caption(f"Room: **{selected_room}** | Updated: {latest['created_at']}")

# ================= METRICS =================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "👥 Total Students",
    latest["total_count"],
    delta=latest["total_count"] - previous["total_count"]
)

c2.metric(
    "📍 Active Zones",
    sum(latest[f"zone{i+1}"] > 0 for i in range(6))
)

avg_util = sum(
    (latest[f"zone{i+1}"] / ZONE_CAPACITY[i]) * 100
    for i in range(6)
) / 6
c3.metric("📊 Avg Utilization", f"{avg_util:.1f}%")

c4.metric(
    "🌡️ Temp (°C)",
    f"{latest_temp:.1f}" if pd.notna(latest_temp) else "—"
)

c5.metric(
    "💧 Humidity (%)",
    f"{latest_hum:.1f}" if pd.notna(latest_hum) else "—"
)

# ================= UTILIZATION =================
st.subheader("⚠️ Zone Utilization & Alerts")

zone_names = [f"Zone {i+1}" for i in range(6)]
zone_occupied = [latest[f"zone{i+1}"] for i in range(6)]
zone_util = [
    (zone_occupied[i] / ZONE_CAPACITY[i]) * 100
    for i in range(6)
]
zone_labels = [
    f"{zone_occupied[i]} / {ZONE_CAPACITY[i]}"
    for i in range(6)
]

for i, util in enumerate(zone_util):
    if util > OVERLOAD_THRESHOLD:
        st.error(f"⚠️ Zone {i+1} overloaded ({util:.1f}%)")

fig_bar = px.bar(
    x=zone_names,
    y=zone_util,
    text=zone_labels,
    labels={"x": "Zone", "y": "Utilization (%)"},
    title="Zone Utilization (%) with Occupied / Capacity",
    range_y=[0, 100]
)

fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ================= FLOOR PLAN =================
st.subheader("🗺️ Floor Plan View")
img = cv2.imread(LAYOUT_IMAGE)
img = cv2.resize(img, (600, 600))
st.image(img, channels="BGR")

# ================= OCCUPANCY TIME SERIES =================
st.subheader("📈 Occupancy Over Time")

fig_occ = px.line(
    df_room.sort_values("created_at"),
    x="created_at",
    y="total_count",
    title="Total Occupancy Trend"
)
st.plotly_chart(fig_occ, use_container_width=True)

# ================= ENVIRONMENT TIME SERIES =================
st.subheader("🌡️ Environment Trends")

fig_temp = px.line(
    df_room.sort_values("created_at"),
    x="created_at",
    y="temperature",
    title="Temperature vs Time"
)

fig_hum = px.line(
    df_room.sort_values("created_at"),
    x="created_at",
    y="humidity",
    title="Humidity vs Time"
)

st.plotly_chart(fig_temp, use_container_width=True)
st.plotly_chart(fig_hum, use_container_width=True)

# ================= COMFORT ALERTS =================
st.subheader("🧠 Comfort Alerts")

if pd.notna(latest_temp) and latest_temp > 28 and latest["total_count"] > 30:
    st.error("🔥 Hot & crowded — ventilation recommended")

if pd.notna(latest_hum) and latest_hum > 70:
    st.warning("💧 High humidity — discomfort likely")

# ================= EXPLAINABILITY =================
with st.expander("ℹ️ How this system works"):
    st.markdown("""
    - Vision-based multi-camera occupancy detection  
    - Homography projection to floor plan  
    - DBSCAN-based duplicate removal  
    - ESP32-based temperature & humidity sensing  
    - NTP-synchronised timestamps  
    - Time-aligned fusion of occupancy & environment data  
    """)
