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
    st.info(
        "No data in selected time range. Showing latest available data instead."
    )
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

# ================= UTILIZATION & ALERTS =================
st.subheader("⚠️ Zone Utilization & Alerts")

zone_util = []
for i in range(6):
    util = (latest[f"zone{i+1}"] / ZONE_CAPACITY[i]) * 100
    zone_util.append(util)
    if util > OVERLOAD_THRESHOLD:
        st.error(f"⚠️ Zone {i+1} overloaded ({util:.1f}%)")

# ================= UTILIZATION BAR =================
fig_bar = px.bar(
    x=[f"Zone {i+1}" for i in range(6)],
    y=zone_util,
    labels={"x": "Zone", "y": "Utilization (%)"},
    title="Zone Utilization (%)",
    range_y=[0, 100]
)
st.plotly_chart(fig_bar, use_container_width=True)

# ================= FLOOR PLAN (HEAT VIEW) =================
st.subheader("🗺️ Floor Plan (Heat View)")

img = cv2.imread(LAYOUT_IMAGE)
img = cv2.resize(img, (600, 600))

def zone_color(p):
    if p < 50:
        return (0, 255, 0)
    elif p < 80:
        return (0, 255, 255)
    else:
        return (0, 0, 255)

# # Adjust these coordinates to your layout
# zones_px = [
#     (30, 30, 200, 200),
#     (220, 30, 400, 200),
#     (420, 30, 580, 200),
#     (30, 220, 200, 580),
#     (220, 220, 400, 580),
#     (420, 220, 580, 580),
# ]

# for i, (x1, y1, x2, y2) in enumerate(zones_px):
#     cv2.rectangle(img, (x1, y1), (x2, y2), zone_color(zone_util[i]), 3)
#     cv2.putText(
#         img,
#         f"Z{i+1}: {latest[f'zone{i+1}']}",
#         (x1 + 10, y1 + 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (255, 255, 255),
#         2
#     )

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
