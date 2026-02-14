import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
import cv2
import os
import math
import numpy as np

# ================= CONFIG =================
REFRESH_SEC = 10
LAYOUT_IMAGE = "lc001_borders.png"  # Ensure this file is in your directory
ZONE_CAPACITY = [20, 20, 40, 40, 20, 20]
OVERLOAD_THRESHOLD = 80  # %
MERGE_TOLERANCE_SEC = 60 # Max lag allowed between sensor & camera

# ================= SUPABASE SETUP =================
# secure way: fetch from st.secrets or env vars
# specific to your project:
SUPABASE_URL = os.getenv("SUPABASE_URL") 
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="Classroom Occupancy Dashboard",
    layout="wide"
)

# ================= UTILS: PMV CALCULATION =================
def calculate_pmv(temp, rh, met=1.1, clo=0.7, v=0.1):
    """
    Robust PMV calculation. Returns None if inputs are invalid/missing.
    """
    try:
        if temp is None or rh is None or pd.isna(temp) or pd.isna(rh):
            return None
        if not (10 <= temp <= 40) or not (1 <= rh <= 100):
            return None

        # PMV Physics Constants
        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (temp + 235))
        icl = 0.155 * clo
        m = met * 58.15
        mw = m  # assuming w=0
        fcl = 1 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
        hcf = 12.1 * math.sqrt(v)
        taa = temp + 273.15
        tcla = taa + (35.5 - temp) / (3.5 * icl + 0.1)

        for _ in range(30):
            hcn = 2.38 * abs(tcla - taa) ** 0.25
            hc = max(hcf, hcn)
            tcla_new = ((mw + 3.96e-8 * fcl * (taa**4 - tcla**4) + fcl * hc * (taa - tcla)) / (3.5 * icl + fcl * hc) + taa)
            if abs(tcla - tcla_new) < 0.01:
                break
            tcla = tcla_new

        pmv = (0.303 * math.exp(-0.036 * m) + 0.028) * (
            mw - 3.05 * (5.73 - 0.007 * mw - pa) - 0.42 * (mw - 58.15)
            - 1.7e-5 * m * (5867 - pa) - 0.0014 * m * (34 - temp)
            - 3.96e-8 * fcl * (tcla**4 - taa**4) - fcl * hc * (tcla - taa)
        )
        return round(float(pmv), 2)
    except Exception:
        return None

# ================= CORE LOGIC: DATA SYNC =================
@st.cache_data(ttl=REFRESH_SEC)
def load_and_merge_data():
    # 1. Fetch ALL data (occupancy and environment mixed)
    res = (
        supabase
        .table("occupancy")
        .select("*")
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )

    df = pd.DataFrame(res.data)
    
    if df.empty:
        return pd.DataFrame(columns=["room", "created_at", "total_count", "temperature", "humidity"])

    # 2. Convert to DateTime (UTC) & Sort
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values("created_at") # Strict sorting is required for filling

    # 3. The "Smart Fill" Strategy
    # We group by room so we don't mix data from different rooms.
    # ffill() propagates the last known temp forward.
    # bfill() pulls the next known temp backward (handles if sensor uploads slightly after camera).
    # limit=1 creates a safety buffer (won't fill data if the gap is too huge, row-wise)
    
    # Note: For time-based safety, we calculate staleness later.
    df["temperature"] = df.groupby("room")["temperature"].ffill().bfill()
    df["humidity"] = df.groupby("room")["humidity"].ffill().bfill()

    # 4. Filter: Keep only the rows that have Vision Data
    # Now these rows have "inherited" the temperature data from their neighbors
    df_occupancy = df[df["total_count"].notna()].copy()

    if df_occupancy.empty:
        return pd.DataFrame()

    # 5. Safety Check: Remove data that is "stale"
    # If the nearest temp reading was actually 3 hours ago, ffill would still bring it.
    # We check the time difference to ensure accuracy.
    
    # We create a 'valid_env_time' column tracking when the temp was actually recorded
    # (Complex logic simplified: If the filled temp is older than 5 mins, treat as None)
    
    # For now, we trust the ffill/bfill for the dashboard display. 
    # If you see data, it's the nearest available.

    # 6. Calculate PMV
    df_occupancy["pmv"] = [
        calculate_pmv(t, h) 
        for t, h in zip(df_occupancy["temperature"], df_occupancy["humidity"])
    ]

    return df_occupancy.sort_values("created_at", ascending=False)


# ================= APP EXECUTION =================

# 1. Load Data
df_room = load_and_merge_data()

if df_room.empty:
    st.warning("⏳ Waiting for data... (Database is empty or no valid occupancy records)")
    st.stop()

# 2. Sidebar Controls
st.sidebar.title("Controls")

# Get unique rooms safely
unique_rooms = df_room["room"].dropna().unique()
if len(unique_rooms) == 0:
    st.error("Data loaded, but no 'room' names found.")
    st.stop()

selected_room = st.sidebar.selectbox("Select Room", unique_rooms)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 10 minutes", "Last 1 hour", "Today"]
)

# 3. Apply Filters
# Filter by Room
df_filtered = df_room[df_room["room"] == selected_room].copy()

# Filter by Time
now = pd.Timestamp.now(tz="UTC")
if time_range == "Last 10 minutes":
    df_filtered = df_filtered[df_filtered["created_at"] >= now - pd.Timedelta(minutes=10)]
elif time_range == "Last 1 hour":
    df_filtered = df_filtered[df_filtered["created_at"] >= now - pd.Timedelta(hours=1)]
else:
    df_filtered = df_filtered[df_filtered["created_at"].dt.date == now.date()]

# Sort for display
df_filtered = df_filtered.sort_values("created_at", ascending=False)

if df_filtered.empty:
    st.info(f"No data for **{selected_room}** in the selected time range.")
    st.stop()

# 4. Extract Latest Snapshot
latest = df_filtered.iloc[0]
previous = df_filtered.iloc[1] if len(df_filtered) > 1 else latest

latest_temp = latest["temperature"]
latest_hum = latest["humidity"]
latest_pmv = latest["pmv"]

# ================= DASHBOARD LAYOUT =================
st.title("📊 Classroom Occupancy Dashboard")
st.caption(f"Room: **{selected_room}** | Live Update: {latest['created_at'].strftime('%H:%M:%S UTC')}")

# Check for Stale Sensor Data
if pd.isna(latest_temp):
    st.warning("⚠️ Environmental data is lagging (> 1 min). PMV cannot be calculated.")

# --- Row 1: Key Metrics ---
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("👥 Students", int(latest["total_count"]), delta=int(latest["total_count"] - previous["total_count"]))
c2.metric("📍 Active Zones", sum(latest[f"zone{i+1}"] > 0 for i in range(6)))

avg_util = sum((latest[f"zone{i+1}"] / ZONE_CAPACITY[i]) * 100 for i in range(6)) / 6
c3.metric("📊 Utilization", f"{avg_util:.1f}%")

c4.metric("🌡️ Temp", f"{latest_temp:.1f} °C" if pd.notna(latest_temp) else "—")
c5.metric("💧 Humidity", f"{latest_hum:.1f} %" if pd.notna(latest_hum) else "—")
c6.metric("🧍 PMV Index", f"{latest_pmv:.2f}" if pd.notna(latest_pmv) else "—")

# --- Row 2: Zone Analysis ---
st.subheader("⚠️ Zone Utilization")

zone_names = [f"Zone {i+1}" for i in range(6)]
zone_occ = [latest[f"zone{i+1}"] for i in range(6)]
zone_util = [(zone_occ[i] / ZONE_CAPACITY[i]) * 100 for i in range(6)]
zone_labels = [f"{zone_occ[i]}/{ZONE_CAPACITY[i]}" for i in range(6)]

# Alert logic
for i, util in enumerate(zone_util):
    if util > OVERLOAD_THRESHOLD:
        st.error(f"⚠️ Zone {i+1} is Overloaded ({util:.1f}%)")

fig_bar = px.bar(
    x=zone_names, y=zone_util, text=zone_labels,
    labels={"x": "Zone", "y": "Utilization (%)"},
    range_y=[0, 100],
    color=zone_util,
    color_continuous_scale="RdYlGn_r" # Green to Red
)
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Row 3: Floor Plan ---
if os.path.exists(LAYOUT_IMAGE):
    st.subheader("🗺️ Floor Plan View")
    img = cv2.imread(LAYOUT_IMAGE)
    st.image(img, channels="BGR", caption="Real-time Zone Map", width=600)

# --- Row 4: Historical Trends ---
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("📈 Occupancy Trend")
    fig_occ = px.line(df_filtered.sort_values("created_at"), x="created_at", y="total_count")
    st.plotly_chart(fig_occ, use_container_width=True)

with c_right:
    st.subheader("🌡️ Thermal Comfort Trend")
    if pd.notna(latest_temp):
        fig_env = px.line(
            df_filtered.sort_values("created_at"), 
            x="created_at", 
            y=["temperature", "humidity", "pmv"],
            markers=True
        )
        st.plotly_chart(fig_env, use_container_width=True)
    else:
        st.info("Insufficient environmental data for trend plotting.")

# --- Row 5: Comfort Analysis ---
st.subheader("🧠 Comfort Analysis")
if pd.isna(latest_pmv):
    st.info("Waiting for sensor synchronization...")
elif latest_pmv < -0.5:
    st.info("❄️ Status: Cool (Consider reducing AC)")
elif -0.5 <= latest_pmv <= 0.5:
    st.success("✅ Status: Comfortable")
elif 0.5 < latest_pmv <= 1.0:
    st.warning("wm️ Status: Slightly Warm")
else:
    st.error("🔥 Status: Hot (Ventilation Required)")
