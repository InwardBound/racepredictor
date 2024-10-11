import gpxpy
import plotly.graph_objs as go
import networkx as nx
import os
import re
import numpy as np
import datetime
import json
from GPXpredictor import remove_graph_points
from app import load_endpoints
import pandas as pd
import glob
def format_time(seconds):
    """Convert time in seconds to hh:mm:ss format."""
    return str(datetime.timedelta(seconds=int(seconds)))
def parse_node_label(label):
    match = re.search(r"trkpt:([-\d.]+),([-\d.]+)@([\d.]+)", label)
    if match:
        lat, lon, elevation = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return lat, lon, elevation
    return None, None, None
def get_closest_node(G, lat, lon):
    closest_node = None
    min_distance = float('inf')
    for node in G.nodes:
        label = G.nodes[node]['label']
        node_lat, node_lon, _ = parse_node_label(label)
        if node_lat is not None and node_lon is not None:
            dist = np.sqrt((lat - node_lat) ** 2 + (lon - node_lon) ** 2)
            if dist < min_distance:
                min_distance = dist
                closest_node = node
    return closest_node
def visualize_optimal_route_graph(G, endpoint, title, dps):
    edge_x = []
    edge_y = []
    edge_color = []
    edge_weight = []

    # Get the node closest to the endpoint (assume it's the endpoint itself for now)
    endpoint_node = None
    min_distance = float('inf')
    for node in G.nodes:
        label = G.nodes[node]['label']
        lat, lon, _ = parse_node_label(label)
        if lat is not None and lon is not None:
            dist = np.sqrt((lat - endpoint.latitude) ** 2 + (lon - endpoint.longitude) ** 2)
            if dist < min_distance:
                min_distance = dist
                endpoint_node = node

    # Find the shortest path time/weight to the endpoint for each node
    shortest_paths = nx.shortest_path_length(G, source=endpoint_node, weight='weight')
    max_weight = max(shortest_paths.values()) if shortest_paths else 1

    # Create node trace
    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []

    for node in G.nodes():
        label = G.nodes[node]['label']
        lat, lon, elevation = parse_node_label(label)
        if lat is not None:
            node_x.append(lon)
            node_y.append(lat)

            # Get shortest path time/weight to the endpoint
            if node in shortest_paths:
                time_to_endpoint = shortest_paths[node]
            else:
                time_to_endpoint = max_weight  # Assign max value if no path is found

            # Normalize the time/weight for scaling
            normalized_time = time_to_endpoint
            node_size.append(5)
            node_color.append(normalized_time)  # Color based on proximity to endpoint
            node_text.append(f"Lat: {lat}, Lon: {lon}, Time to endpoint: {time_to_endpoint}")

    node_trace = go.Scattermapbox(
        lon=node_x,
        lat=node_y,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale='rdylgn',  # Use a heatmap color scale (green to red)
            reversescale=True,
            colorbar=dict(
                title="Time to Endpoint (assuming 2m/s Gradient Adjusted Speed)",  # Label for the color scal
                titleside='right',
            ),
            showscale=True
        ),
        text=node_text,
        hoverinfo='text'
    )

    # Create the figure
    fig = go.Figure(data=[node_trace])
    fig.update_layout(
        mapbox_style="open-street-map",
        title = title,
        mapbox=dict(
            center=dict(lat=endpoint.latitude, lon=endpoint.longitude),  # Access endpoint attributes
            zoom=9
        ),
        showlegend=False
    )


    folder_path = os.path.join(os.getcwd(),  "graphs", race_slug)
    fig.write_html(os.path.join(folder_path, title + ".html"))


def calculate_combined_shortest_path(G, start_node, end_node):
    # Calculate shortest path lengths (time/weight) from start node to all other nodes
    shortest_paths_from_start = nx.shortest_path_length(G, source=start_node, weight='weight')

    # Calculate shortest path lengths (time/weight) from all other nodes to the end node
    shortest_paths_to_end = nx.shortest_path_length(G, source=end_node, weight='weight')

    combined_times = {}

    # For each node, sum the time from start to the node and from the node to the end
    for node in G.nodes():
        time_from_start = shortest_paths_from_start.get(node, float('inf'))
        time_to_end = shortest_paths_to_end.get(node, float('inf'))
        combined_times[node] = time_from_start + time_to_end

    return combined_times


# Function to visualize the graph with nodes colored based on the combined shortest path time
def visualize_combined_shortest_path(G,  endpoint,start_point,title, race_slug):
    # Get the closest nodes to the start point and the endpoint
    start_node = get_closest_node(G, start_point.latitude, start_point.longitude)
    end_node = get_closest_node(G, endpoint.latitude, endpoint.longitude)

    # Calculate the combined shortest path time (start -> node -> end) for each node
    combined_times = calculate_combined_shortest_path(G, start_node, end_node)
    max_combined_time = max(combined_times.values()) if combined_times else 1

    # Create node trace
    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []

    for node in G.nodes():
        label = G.nodes[node]['label']
        lat, lon, elevation = parse_node_label(label)
        if lat is not None:
            node_x.append(lon)
            node_y.append(lat)

            # Get the combined shortest path time (start -> node -> end)
            combined_time = combined_times.get(node, max_combined_time)

            # Normalize the time for scaling
            node_size.append(5)
            node_color.append(combined_time)  # Color based on combined time
            node_text.append(f"Lat: {lat}, Lon: {lon}, Combined Time: {combined_time}")

    # Determine the shortest and capped maximum time
    shortest_time = min(node_color)
    capped_max_time = 1.2 * shortest_time

    # Apply the cap to node_color values
    node_color_capped = [min(time, capped_max_time) for time in node_color]

    # Apply logarithmic transformation to node_color values (for visualization)
    node_color_log = np.log(np.array(node_color_capped) + 1e-5)  # Apply log transformation

    # Define the tick values and corresponding text for the colorbar (in hh:mm:ss)
    num_ticks = 5
    tickvals = np.log(np.linspace(shortest_time, capped_max_time, num_ticks) + 1e-5)
    ticktext = [format_time(val) for val in np.linspace(shortest_time, capped_max_time, num_ticks)]

    node_trace = go.Scattermapbox(
        lon=node_x,
        lat=node_y,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color_log,  # Use the log-transformed values for coloring
            colorscale='rdylgn',  # Use a heatmap color scale (green to red)
            reversescale=True,  # Reverse the color scale
            colorbar=dict(
                title="Combined Time (Start -> Node -> End)",  # Actual time in the title
                titleside='right',
                tickvals=tickvals,  # Logarithmic ticks
                ticktext=ticktext,  # Display times in hh:mm:ss format
            ),
            showscale=True
        ),
        text=node_text,
        hoverinfo='text'
    )
    drop_trace = go.Scattermapbox(
        lon=[start_point.longitude],
        lat=[start_point.latitude],
        mode='markers+text',
        marker=dict(
            size=10,
            color='blue',  # Color for drop point
            symbol='circle'
        ),
        text=["Division Drop Point"],
        textposition="top center",
        hoverinfo='text'
    )

    # Mark the endpoint with a special icon or symbol (e.g., flag)
    endpoint_trace = go.Scattermapbox(
        lon=[endpoint.longitude],
        lat=[endpoint.latitude],
        mode='markers+text',
        marker=dict(
            size=12,
            color='red',  # Color for endpoint
            symbol='flag'  # Use flag symbol for finish
        ),
        text=["Finish"],
        textposition="top right",
        hoverinfo='text'
    )

    # Create the figure
    fig = go.Figure(data=[node_trace,drop_trace,endpoint_trace])
    fig.update_layout(
        title_text=title,
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=np.average([endpoint.latitude, start_point.latitude]), lon=np.average([start_point.longitude,endpoint.longitude])),
            zoom=10
        ),
        showlegend=False
    )

    folder_path = os.path.join(os.getcwd(), "graphs", race_slug)
    fig.write_html(os.path.join(folder_path,title+".html"))

race_slug = '2024'
folder_path = os.path.join(os.getcwd(), "vertdata")
graphcache = 'graph' + race_slug + '.gexf'
gpxcache = 'routes' + race_slug + '.gpx'
graph_path = os.path.join(folder_path, graphcache)
G = nx.read_gexf(graph_path)
endpoints_file_path = os.path.join(os.getcwd(), 'endpoints.txt')
endpoints = load_endpoints(endpoints_file_path)
drops_folder_path = os.path.join(os.getcwd(), 'droppoints')
drops_file_path = os.path.join(drops_folder_path,race_slug + ".csv")
drop_df = pd.read_csv(drops_file_path)
print(endpoints)
lat, long, alt = endpoints[race_slug]
endpoint = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=long,elevation=alt)
visualize_optimal_route_graph(G, endpoint, 'time to EP',drop_df)
ex_folder_path = os.path.join(os.getcwd(), "exclusions", race_slug)

cps=[]
print(drop_df)
for divno in range(1,8):
    divex_files = glob.glob(os.path.join(ex_folder_path, f'*{divno}*.geojson'))
    divex_graph_path = os.path.join(ex_folder_path, str(divno) + ".gexf")
    div_route_graph = G.copy()
    dp= gpxpy.gpx.GPXTrackPoint(latitude=drop_df['lat'][divno-1], longitude=drop_df['lon'][divno-1])
    if divex_files:
        if os.path.exists(divex_graph_path):
            div_route_graph = nx.read_gexf(divex_graph_path)
        else:
            for divex_file in divex_files:
                print(divex_file)
                with open(divex_file, 'r') as file:
                    areas_data = json.load(file)
                div_route_graph = remove_graph_points(div_route_graph, areas_data)
            nx.write_gexf(div_route_graph, divex_graph_path)
    visualize_combined_shortest_path(div_route_graph,endpoint,dp,'Division ' +str(divno) + ' Optimal Route Map',race_slug)
    visualize_optimal_route_graph(div_route_graph,dp,'Division ' + str(divno) + ' Time from drop',drop_df)
"""
for cp in cps:
    for divno in range(1, 8):
        divex_files = glob.glob(os.path.join(ex_folder_path, f'*{divno}*.geojson'))
        divex_graph_path = os.path.join(ex_folder_path, str(divno) + ".gexf")
        div_route_graph = G.copy()
        if divex_files:
            if os.path.exists(divex_graph_path):
                div_route_graph = nx.read_gexf(divex_graph_path)
            else:
                for divex_file in divex_files:
                    print(divex_file)
                    with open(divex_file, 'r') as file:
                        areas_data = json.load(file)
                    div_route_graph = remove_graph_points(div_route_graph, areas_data)
                nx.write_gexf(div_route_graph, divex_graph_path)
        visualize_combined_shortest_path(div_route_graph, gpxpy.gpx.GPXTrackPoint(latitude=cp[0], longitude=cp[1]),
                                         gpxpy.gpx.GPXTrackPoint(latitude=drop_df['lat'][divno - 1],
                                                                 longitude=drop_df['lon'][divno - 1]),
                                         'Division ' + str(divno) + ' Optimal Route Map to ' + str(cp))

"""