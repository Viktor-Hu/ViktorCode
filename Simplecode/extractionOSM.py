import osmnx as ox
import overpy
import geopandas as gpd
from shapely.geometry import Point

def get_traffic_signs_in_area(bbox):
    api = overpy.Overpass()
    query = f"""
    [out:json];
    (
        node["traffic_sign"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out body;
    """
    response = api.query(query)
    return response.nodes

city = "Arlington"
state = "TX"
country = "USA"

city_graph = ox.graph_from_place(f'{city}, {state}, {country}', network_type='drive')
nodes_gdf, edges_gdf = ox.graph_to_gdfs(city_graph)
nodes_gdf_wgs84 = nodes_gdf.to_crs(epsg=4326)
city_bbox = nodes_gdf_wgs84.total_bounds

traffic_signs = get_traffic_signs_in_area(city_bbox)

traffic_sign_data = []

for traffic_sign in traffic_signs:
    sign_info = {
        "id": traffic_sign.id,
        "tags": traffic_sign.tags,
        "geometry": Point(float(traffic_sign.lon), float(traffic_sign.lat))
    }
    traffic_sign_data.append(sign_info)

gdf = gpd.GeoDataFrame(traffic_sign_data, columns=["id", "tags", "geometry"], crs="EPSG:4326")
gdf.to_file("traffic_signs_with_attributes.geojson", driver="GeoJSON")
