import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d, Dropout, LayerNorm
import holidays
import pickle
from datetime import datetime, timedelta, time
from math import sin, cos, pi, radians, atan2, sqrt
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import requests

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.spatial_proj = Linear(in_channels, hidden_channels)
        self.tcn = Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(out_channels)
        self.dropout = Dropout(0.1)
        if in_channels != out_channels:
            self.residual_conv = Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x, norm_adj_sparse):
        batch_size, _, seq_len, in_channels = x.shape

        # Residual path
        x_res_in = x.permute(0, 1, 3, 2).reshape(-1, in_channels, seq_len)
        if self.residual_conv:
            x_residual = self.residual_conv(x_res_in)
        else:
            x_residual = x_res_in
        x_residual = x_residual.reshape(batch_size, self.num_nodes, self.out_channels, seq_len)
        x_residual = x_residual.permute(0, 1, 3, 2)  # (B, N, T, out_channels)

        # Spatial GCN
        x_gcn_in = x.permute(0, 2, 1, 3).reshape(-1, self.num_nodes, in_channels)
        x_proj_in = x_gcn_in.reshape(-1, in_channels)
        x_proj_out = self.spatial_proj(x_proj_in)
        x_conv_in = x_proj_out.reshape(-1, self.num_nodes, self.hidden_channels)

        gcn_out_list = []
        for i in range(x_conv_in.shape[0]):
            snapshot = x_conv_in[i]
            gcn_out = torch.sparse.mm(norm_adj_sparse, snapshot)
            gcn_out_list.append(gcn_out)

        x_gcn_out = torch.stack(gcn_out_list)
        x_gcn_out = x_gcn_out.reshape(batch_size, seq_len, self.num_nodes, self.hidden_channels)
        x_gcn_out = self.norm1(x_gcn_out)
        x_gcn_out = F.relu(x_gcn_out)
        x_gcn_out = x_gcn_out.permute(0, 2, 1, 3)

        # Temporal Conv
        x_tcn_in = x_gcn_out.permute(0, 1, 3, 2).reshape(-1, self.hidden_channels, seq_len)
        x_tcn_out = self.tcn(x_tcn_in)
        x_tcn_out = x_tcn_out.reshape(batch_size, self.num_nodes, self.out_channels, seq_len)
        x_tcn_out = x_tcn_out.permute(0, 1, 3, 2)
        x_tcn_out = self.norm2(x_tcn_out)
        x_tcn_out = F.relu(x_tcn_out)

        # Residual + dropout
        out = x_tcn_out + x_residual
        out = self.dropout(out)
        return out


class STGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_features, seq_len):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(num_features, 64, 32, num_nodes)
        self.block2 = STGCNBlock(64, 128, 64, num_nodes)
        self.final_linear1 = Linear(128, 64)
        self.final_linear2 = Linear(64, 1)
        self.dropout_out = Dropout(0.1)

    def forward(self, x, norm_adj_sparse):
        x = self.block1(x, norm_adj_sparse)
        x = self.block2(x, norm_adj_sparse)
        x = x[:, :, -1, :]  # last time step
        x = F.relu(self.final_linear1(x))
        x = self.dropout_out(x)
        x = self.final_linear2(x)
        return x


# ==========================================
# 2. PREDICTOR LOGIC
# ==========================================
class TrafficPredictor:
    def __init__(self):
        self.device = torch.device("cpu")

        try:
            self.profile_df = pd.read_pickle("traffic_profile.pkl").set_index(
                ["GEOHASH", "day_of_week", "hour"]
            )
            self.static_df = pd.read_pickle("sensor_static_data.pkl")
            self.sensor_order = self.static_df["GEOHASH"].values
            self.num_sensors = len(self.sensor_order)

            with open("scalers.pkl", "rb") as f:
                self.scalers = pickle.load(f)

            self.adj_matrix = torch.load("adj_matrix.pt", map_location=self.device)

            self.model = STGCN(
                num_nodes=self.num_sensors, num_features=20, seq_len=12
            ).to(self.device)
            self.model.load_state_dict(
                torch.load("best_stgcn_model_traffic_flow.pth", map_location=self.device)
            )
            self.model.eval()

            self.tr_holidays = holidays.Turkey()
        except FileNotFoundError as e:
            st.error(f"Missing file: {e}. Please run prepare_data.py first.")
            st.stop()

    def get_temporal_features(self, dt):
        hour_float = dt.hour + dt.minute / 60.0
        return {
            "hour_sin": sin(2 * pi * hour_float / 24.0),
            "hour_cos": cos(2 * pi * hour_float / 24.0),
            "day_of_week_sin": sin(2 * pi * dt.weekday() / 7.0),
            "day_of_week_cos": cos(2 * pi * dt.weekday() / 7.0),
            "is_weekend": 1 if dt.weekday() >= 5 else 0,
            "is_rush_hour": 1 if (7 <= dt.hour <= 9) or (17 <= dt.hour <= 19) else 0,
            "is_holiday": 1 if dt in self.tr_holidays else 0,
        }

    def predict(self, target_date, target_time, weather_data):
        full_dt = datetime.combine(target_date, target_time)
        sequence_timestamps = [full_dt - timedelta(hours=i) for i in range(12, 0, -1)]

        dynamic_features = []
        for ts in sequence_timestamps:
            try:
                step_profile = self.profile_df.loc[
                    (slice(None), ts.weekday(), ts.hour), :
                ].reset_index()
                step_profile = (
                    step_profile.set_index("GEOHASH")
                    .reindex(self.sensor_order)
                    .fillna(0)
                )
                speeds = step_profile["AVERAGE_SPEED"].values
                vehicles = step_profile["NUMBER_OF_VEHICLES"].values
            except KeyError:
                speeds = np.zeros(self.num_sensors)
                vehicles = np.zeros(self.num_sensors)

            time_feats = self.get_temporal_features(ts)

            step_matrix = np.column_stack(
                [
                    speeds,
                    vehicles,
                    np.full(self.num_sensors, time_feats["hour_sin"]),
                    np.full(self.num_sensors, time_feats["hour_cos"]),
                    np.full(self.num_sensors, time_feats["day_of_week_sin"]),
                    np.full(self.num_sensors, time_feats["day_of_week_cos"]),
                    np.full(self.num_sensors, time_feats["is_weekend"]),
                    np.full(self.num_sensors, time_feats["is_rush_hour"]),
                    np.full(self.num_sensors, time_feats["is_holiday"]),
                    np.full(self.num_sensors, weather_data["temperature"]),
                    np.full(self.num_sensors, weather_data["humidity"]),
                    np.full(self.num_sensors, weather_data["precipitation"]),
                    np.full(self.num_sensors, weather_data["is_raining"]),
                ]
            )
            dynamic_features.append(step_matrix)

        X_dynamic = np.stack(dynamic_features, axis=1)
        X_static = self.static_df.drop(columns=["GEOHASH"]).values

        N, T, F_dyn = X_dynamic.shape
        X_dyn_scaled = self.scalers["dyn"].transform(
            X_dynamic.reshape(-1, F_dyn)
        ).reshape(N, T, F_dyn)
        X_stat_scaled = self.scalers["stat"].transform(X_static)

        X_dyn_tensor = torch.FloatTensor(X_dyn_scaled).to(self.device)
        X_stat_tensor = torch.FloatTensor(X_stat_scaled).to(self.device)
        X_stat_expanded = X_stat_tensor.unsqueeze(1).expand(-1, T, -1)
        X_input = torch.cat([X_dyn_tensor, X_stat_expanded], dim=2).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = self.model(X_input, self.adj_matrix)

        prediction_actual = self.scalers["y"].inverse_transform(
            prediction_scaled.cpu().numpy().reshape(-1, 1)
        )

        return pd.DataFrame(
            {
                "GEOHASH": self.sensor_order,
                "LATITUDE": self.static_df["LATITUDE"].values,
                "LONGITUDE": self.static_df["LONGITUDE"].values,
                "TRAFFIC_FLOW": prediction_actual.flatten(),
            }
        )

# ==========================================
# 2.5 GEOCODING + ROUTING HELPERS
# ==========================================
@st.cache_resource
def get_geocode_func():
    geolocator = Nominatim(user_agent="istanbul-traffic-app")
    return RateLimiter(geolocator.geocode, min_delay_seconds=1)

def geocode_address(address: str):
    geocode = get_geocode_func()
    location = geocode(address)
    if location is None:
        return None, None
    return location.latitude, location.longitude

def get_route_osrm(start_lon, start_lat, end_lon, end_lat):
    """
    Query OSRM demo server for a driving route.
    Returns list of (lat, lon) along the route, or None on failure.
    """
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {"overview": "full", "geometries": "geojson"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        routes = data.get("routes")
        if not routes:
            return None
        coords = routes[0]["geometry"]["coordinates"]  # [lon, lat]
        return [(lat, lon) for lon, lat in coords]
    except Exception as e:
        st.warning(f"Could not get OSRM route: {e}")
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_route_congestion_stats(results_df, route_coords, max_distance_km=0.5):
    """
    Find sensors within max_distance_km of the route polyline
    and compute basic congestion stats over their predicted TRAFFIC_FLOW.
    """
    if route_coords is None or len(route_coords) == 0:
        return None

    sensor_lats = results_df["LATITUDE"].values
    sensor_lons = results_df["LONGITUDE"].values
    flows = results_df["TRAFFIC_FLOW"].values

    mask = np.zeros(len(sensor_lats), dtype=bool)
    # Subsample route to at most ~100 points for speed
    sampled_route = route_coords[:: max(1, len(route_coords) // 100)]

    for i in range(len(sensor_lats)):
        lat = sensor_lats[i]
        lon = sensor_lons[i]
        dmin = float("inf")
        for (r_lat, r_lon) in sampled_route:
            d = haversine_km(lat, lon, r_lat, r_lon)
            if d < dmin:
                dmin = d
            if dmin <= max_distance_km:
                break
        if dmin <= max_distance_km:
            mask[i] = True

    if not mask.any():
        return None

    route_flows = flows[mask]
    return {
        "num_sensors": int(mask.sum()),
        "avg_flow": float(route_flows.mean()),
        "max_flow": float(route_flows.max()),
    }

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Istanbul Traffic AI", layout="wide")

st.title("🚗 Istanbul Traffic Prediction System")
st.markdown("STGCN Model with Spatio-Temporal Features")

# --- Sidebar: Inputs ---
st.sidebar.header("1. Select Date & Time")
target_date = st.sidebar.date_input("Date", value=datetime.now() + timedelta(days=1))

hour_select = st.sidebar.slider("Select Hour", 0, 23, 14)
minute_select = st.sidebar.slider("Select Minute", 0, 59, 30)
target_time = time(hour_select, minute_select)

st.sidebar.header("2. Weather Conditions")
weather_cond = st.sidebar.selectbox("Condition", ["Clear/Cloudy", "Rainy", "Heavy Rain"])
temp = st.sidebar.slider("Temperature (°C)", -5, 40, 20)
humid = st.sidebar.slider("Humidity (%)", 0, 100, 60)

is_raining = 1 if "Rain" in weather_cond else 0
precip = 0.0
if weather_cond == "Rainy":
    precip = 2.0
elif weather_cond == "Heavy Rain":
    precip = 10.0

weather_data = {
    "temperature": temp,
    "humidity": humid,
    "precipitation": precip,
    "is_raining": is_raining,
}

# --- Route selection (OPTIONAL) ---
st.sidebar.header("3. Route (optional)")

enable_route = st.sidebar.checkbox("Show route between two locations")

route_input_type = st.sidebar.radio(
    "Location input type",
    ["Coordinates", "Address / Place name"],
    index=1 if enable_route else 0,
)

start_lat = start_lon = end_lat = end_lon = None
start_address = end_address = ""

if enable_route:
    if route_input_type == "Coordinates":
        st.sidebar.markdown("**Start location (lat, lon)**")
        start_lat = st.sidebar.number_input(
            "Start latitude",
            min_value=40.7,
            max_value=41.4,
            value=41.0082,
            step=0.0001,
            key="start_lat",
        )
        start_lon = st.sidebar.number_input(
            "Start longitude",
            min_value=28.4,
            max_value=29.6,
            value=28.9784,
            step=0.0001,
            key="start_lon",
        )

        st.sidebar.markdown("**End location (lat, lon)**")
        end_lat = st.sidebar.number_input(
            "End latitude",
            min_value=40.7,
            max_value=41.4,
            value=41.05,
            step=0.0001,
            key="end_lat",
        )
        end_lon = st.sidebar.number_input(
            "End longitude",
            min_value=28.4,
            max_value=29.6,
            value=29.02,
            step=0.0001,
            key="end_lon",
        )
    else:
        # Address / place mode
        st.sidebar.markdown("**Start location (address / place)**")
        start_address = st.sidebar.text_input(
            "Start",
            value="Taksim Square, Istanbul",
            key="start_address",
        )

        st.sidebar.markdown("**End location (address / place)**")
        end_address = st.sidebar.text_input(
            "End",
            value="Istanbul Airport",
            key="end_address",
        )

# --- Session state for predictions ---
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

@st.cache_resource
def load_predictor():
    return TrafficPredictor()

# Button to trigger prediction
if st.sidebar.button("Predict Traffic Flow"):
    with st.spinner("Calculating spatio-temporal graph interactions..."):
        predictor = load_predictor()
        st.session_state["results"] = predictor.predict(
            target_date, target_time, weather_data
        )
        st.session_state["prediction_done"] = True

# --- Display results ---
if st.session_state["prediction_done"]:
    results = st.session_state["results"]

    avg_flow = results["TRAFFIC_FLOW"].mean()
    max_flow = results["TRAFFIC_FLOW"].max()

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Time", f"{target_time.strftime('%H:%M')}")
    c2.metric("Avg Traffic Flow (city)", f"{avg_flow:.1f}")
    c3.metric("Max Congestion (city)", f"{max_flow:.1f}")

    st.subheader("Traffic Heatmap")

    # Base map
    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10, tiles="CartoDB positron")

    # Heatmap
    heat_data = results[["LATITUDE", "LONGITUDE", "TRAFFIC_FLOW"]].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=12).add_to(m)

    route_coords = None
    route_stats = None

    if enable_route:
        s_lat = s_lon = e_lat = e_lon = None

        # Get start/end coords
        if route_input_type == "Coordinates":
            if all(v is not None for v in [start_lat, start_lon, end_lat, end_lon]):
                s_lat, s_lon, e_lat, e_lon = start_lat, start_lon, end_lat, end_lon
        else:
            if start_address.strip():
                s_lat, s_lon = geocode_address(start_address + ", Istanbul, Turkey")
            if end_address.strip():
                e_lat, e_lon = geocode_address(end_address + ", Istanbul, Turkey")
            if s_lat is None or e_lat is None:
                st.warning(
                    "Could not geocode one or both addresses. Please check the names."
                )

        # Route + markers
        if None not in (s_lat, s_lon, e_lat, e_lon):
            start_point = (s_lat, s_lon)
            end_point = (e_lat, e_lon)

            # Get real-road route from OSRM; fallback to straight line
            route_coords = get_route_osrm(s_lon, s_lat, e_lon, e_lat)
            if not route_coords:
                route_coords = [start_point, end_point]

            folium.Marker(
                location=start_point,
                popup=f"Start: {start_address or f'{s_lat:.5f}, {s_lon:.5f}'}",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(m)

            folium.Marker(
                location=end_point,
                popup=f"End: {end_address or f'{e_lat:.5f}, {e_lon:.5f}'}",
                icon=folium.Icon(color="red", icon="stop"),
            ).add_to(m)

            folium.PolyLine(locations=route_coords, weight=5, opacity=0.9).add_to(m)

            # Compute congestion stats along the route
            route_stats = get_route_congestion_stats(
                results, route_coords, max_distance_km=0.5
            )

    st_folium(m, width=None, height=500, key="traffic_map")

    # Route congestion summary
    if enable_route and route_stats is not None:
        city_mean = avg_flow
        route_mean = route_stats["avg_flow"]
        ratio = route_mean / (city_mean + 1e-6)
        # Simple 0–100 score: 50 means "same as city avg", >50 more congested
        score = float(np.clip(ratio * 50, 0, 100))

        st.subheader("Route Congestion Summary")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Sensors near route", route_stats["num_sensors"])
        rc2.metric("Route avg flow", f"{route_mean:.1f}")
        rc3.metric("Route congestion score", f"{score:.0f} / 100")

    elif enable_route and route_stats is None:
        st.info(
            "No sensors found close to this route (within 500m). "
            "Try a different route or increase the search radius in code."
        )

    with st.expander("View Raw Prediction Data"):
        st.dataframe(results)
