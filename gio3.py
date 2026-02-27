import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import pydeck as pdk
import matplotlib.pyplot as plt
import networkx as nx

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(layout="wide")
st.title("🌍 GFIS geospatial")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙ Optimization Settings")

HUB_CAPACITY = st.sidebar.slider("Hub Capacity (TPD)", 1000, 100000, 20000)
carbon_price = st.sidebar.number_input("Carbon Price (₹ / ton CO2)", value=1500)
infra_cost_per_hub = st.sidebar.number_input("Infra Cost per Hub (₹)", value=5000000)

uploaded_file = st.file_uploader("Upload Village Dataset CSV", type=["csv"])

# =====================================================
# MAIN
# =====================================================
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None)

    if lat_col is None or lon_col is None:
        st.error("Latitude or Longitude column not found.")
        st.stop()

    df = df.rename(columns={lat_col: "Latitude", lon_col: "Longitude"})

    if "Village_Name" not in df.columns:
        df["Village_Name"] = "Village_" + (df.index + 1).astype(str)

    if "District" not in df.columns:
        df["District"] = "District_1"

    if "Total_Waste_kg_day" not in df.columns:
        df["Total_Waste_kg_day"] = 1000

    df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    # =====================================================
    # WASTE MODEL
    # =====================================================
    df["Organic_Waste_TPD"] = df["Total_Waste_kg_day"] / 1000
    df["Methane_m3"] = df["Organic_Waste_TPD"] * 100
    df["CO2_ton"] = (df["Methane_m3"] * 0.67) / 1000

    # =====================================================
    # HUB OPTIMIZATION
    # =====================================================
    final_hubs = []
    df["Cluster"] = -1
    cluster_counter = 0

    for district in df["District"].unique():

        district_data = df[df["District"] == district].copy()
        total_waste = district_data["Organic_Waste_TPD"].sum()
        k = max(1, int(np.ceil(total_waste / HUB_CAPACITY)))

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        district_data["Cluster"] = kmeans.fit_predict(
            district_data[["Latitude", "Longitude"]]
        )

        centers = kmeans.cluster_centers_

        for idx, center in enumerate(centers):

            cluster_id = cluster_counter
            df.loc[district_data.index[district_data["Cluster"] == idx], "Cluster"] = cluster_id

            final_hubs.append({
                "Hub_ID": f"{district}_Hub_{idx+1}",
                "District": district,
                "Latitude": center[0],
                "Longitude": center[1],
                "Cluster": cluster_id
            })

            cluster_counter += 1

    hubs_df = pd.DataFrame(final_hubs)

    # =====================================================
    # FINANCIALS
    # =====================================================
    total_carbon_revenue = df["CO2_ton"].sum() * carbon_price
    total_infra_cost = len(hubs_df) * infra_cost_per_hub
    total_profit = total_carbon_revenue - total_infra_cost

    st.subheader("📊 Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Hubs", len(hubs_df))
    c2.metric("Carbon Revenue (₹)", round(total_carbon_revenue))
    c3.metric("Net Profit (₹)", round(total_profit))

    # =====================================================
    # MAP CENTER
    # =====================================================
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()

    # =====================================================
    # MAP 1 – NORMAL MAP
    # =====================================================
    st.subheader("🗺 Normal GIS Map")

    normal_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles="CartoDB positron"
    )

    village_cluster = MarkerCluster(name="Villages").add_to(normal_map)
    hub_cluster = MarkerCluster(name="Hubs").add_to(normal_map)

    heat_data = []

    for _, row in df.iterrows():

        heat_data.append([
            row["Latitude"],
            row["Longitude"],
            row["Organic_Waste_TPD"]
        ])

        popup = f"""
        <b>Village:</b> {row['Village_Name']}<br>
        <b>District:</b> {row['District']}<br>
        <b>Methane:</b> {round(row['Methane_m3'],2)} m³<br>
        <b>CO₂:</b> {round(row['CO2_ton'],3)} ton
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            popup=popup,
            color="blue",
            fill=True
        ).add_to(village_cluster)

    HeatMap(heat_data, radius=15).add_to(normal_map)

    # Hubs + Network
    centers = hubs_df[["Latitude", "Longitude"]].values
    G = nx.Graph()

    for i, hub in hubs_df.iterrows():

        hub_villages = df[df["Cluster"] == hub["Cluster"]]
        total_methane = hub_villages["Methane_m3"].sum()
        total_co2 = hub_villages["CO2_ton"].sum()

        hub_popup = f"""
        <b>Hub Name:</b> {hub['Hub_ID']}<br>
        <b>District:</b> {hub['District']}<br>
        <b>Total Methane:</b> {round(total_methane,2)} m³<br>
        <b>Total CO₂:</b> {round(total_co2,2)} ton
        """

        folium.Marker(
            location=[hub["Latitude"], hub["Longitude"]],
            popup=hub_popup,
            icon=folium.Icon(color="red", icon="industry", prefix="fa")
        ).add_to(hub_cluster)

    if len(centers) > 1:

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):

                distance = geodesic(
                    (centers[i][0], centers[i][1]),
                    (centers[j][0], centers[j][1])
                ).km

                G.add_edge(i, j, weight=distance)

        mst = nx.minimum_spanning_tree(G)

        for edge in mst.edges():
            i, j = edge
            folium.PolyLine(
                [
                    [centers[i][0], centers[i][1]],
                    [centers[j][0], centers[j][1]]
                ],
                color="yellow",
                weight=3
            ).add_to(normal_map)

    st_folium(normal_map, width=1400, height=550)

    # =====================================================
    # MAP 2 – SATELLITE MAP (FAST)
    # =====================================================
    st.subheader("🛰 Satellite GIS Map")

    satellite_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite"
    ).add_to(satellite_map)

    for _, hub in hubs_df.iterrows():

        folium.Marker(
            location=[hub["Latitude"], hub["Longitude"]],
            icon=folium.Icon(color="red", icon="industry", prefix="fa"),
            popup=f"<b>{hub['Hub_ID']}</b><br>District: {hub['District']}"
        ).add_to(satellite_map)

    st_folium(satellite_map, width=1400, height=550)

    # =====================================================
    # AI FORECAST
    # =====================================================
    st.subheader("🧠 AI Waste Forecast")

    current_total = df["Organic_Waste_TPD"].sum()

    years = np.array([1,2,3,4,5]).reshape(-1,1)
    waste = np.array([
        current_total*0.8,
        current_total*0.9,
        current_total,
        current_total*1.05,
        current_total*1.1
    ])

    model = LinearRegression()
    model.fit(years, waste)
    future_pred = model.predict(years)

    fig, ax = plt.subplots()
    ax.plot(years, waste)
    ax.plot(years, future_pred, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Waste (TPD)")
    st.pyplot(fig)

    # =====================================================
    # 3D GIS
    # =====================================================
    st.subheader("🌐 3D GIS Waste Density")

    df["Elevation"] = df["Organic_Waste_TPD"] * 100

    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position='[Longitude, Latitude]',
        get_elevation="Elevation",
        elevation_scale=10,
        radius=2000,
        get_fill_color='[0, 128, 255, 180]',
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=6,
        pitch=50,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Village:</b> {Village_Name}<br>"
                    "<b>District:</b> {District}<br>"
                    "<b>Waste:</b> {Organic_Waste_TPD} TPD"
        },
    )

    st.pydeck_chart(deck)

else:
    st.info("Please upload a village dataset CSV to begin.")
