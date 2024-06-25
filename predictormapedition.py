import os
import gpxpy.gpx
import gpxpy
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import requests
import re
import asyncio
import aiohttp
import networkx as nx
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point
import sys
import pprint
from pmtiles.reader import Reader, MmapSource
def SpdtoGAP(speed, grade):
    multiplier = 14.636*grade*grade+2.8609*grade+1
    return (speed*multiplier)
def GAPtoSPD(GAP, grade):
    divisor = 14.636 * grade * grade + 2.8609 * grade + 1
    return (GAP/divisor)
# Function to fetch road data from OSM based on a bounding box
def fetch_road_data_from_osm(bbox):
    tags = {
        'highway': True,   # Include highways
        'path': True,      # Include paths
        'footway': True,   # Include footways
        'cycleway': True,  # Include cycleways
        'service': True,   # Include service roads
        'track': True      # Include tracks
    }

    # Fetch OSM data within the bounding box
    graph = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type='all')

    return graph

# Function to create a graph from road data
def create_graph_from_road_data(road_data):
    G = nx.Graph()
    for u, v, data in road_data.edges(data=True):
        G.add_edge(u, v, length=data['length'])
    return G

async def fetch_elevations_batch_async(points):
    url = "https://api.opentopodata.org/v1/aster30m?locations="

    # Split points into batches of 100
    batched_points = [points[i:i + 100] for i in range(0, len(points), 100)]

    all_elevations = []

    async with aiohttp.ClientSession() as session:
        for batch in batched_points:
            batch_url = url + "|".join(f"{lat},{lon}" for lat, lon in batch)
            try:
                async with session.get(batch_url) as response:
                    if response.status != 200:
                        print(f"Error fetching elevation data: HTTP {response.status}")
                        continue

                    data = await response.json()
                    if "results" in data and len(data["results"]) > 0:
                        elevations = [result["elevation"] for result in data["results"]]
                        all_elevations.extend(elevations)
                    else:
                        print("No elevation data found in API response")

            except aiohttp.ClientError as e:
                print(f"HTTP request error: {e}")
            except Exception as e:
                print(f"Error fetching elevation data: {e}")

    return all_elevations

async def points_to_gpx(points_data, track_name):
    gpx = gpxpy.gpx.GPX()
    # Create a track with the specified name
    track = gpxpy.gpx.GPXTrack(name=track_name)
    gpx.tracks.append(track)
    # Create a segment for the track
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    # Extract coordinates and timestamps from points_data
    coordinates = [(float(point['lat']), float(point['long'])) for point in points_data]
    timestamps = [datetime.strptime(point['time'], '%Y-%m-%dT%H:%M:%S.%fZ') for point in points_data]

    # Sort points_data based on timestamps (from earliest to latest)
    points_data_sorted = sorted(points_data, key=lambda x: datetime.strptime(x['time'], '%Y-%m-%dT%H:%M:%S.%fZ'))

    # Fetch elevations asynchronously in batches
    elevations = await fetch_elevations_batch_async(coordinates)

    # Assign elevations to points and add to GPX track segment in the sorted order
    for point, elevation in zip(points_data_sorted, elevations):
        lat = float(point['lat'])
        lon = float(point['long'])
        time = datetime.strptime(point['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
        gpx_point = gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=float(elevation), time=time)
        segment.points.append(gpx_point)

    return gpx

# Function to find the optimal path using the GAP model
def find_optimal_path_gap(start_point, end_point, graph):
    start_coords = (start_point.longitude, start_point.latitude)
    end_coords = (end_point.longitude, end_point.latitude)
    start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
    shortest_path_nodes = nx.shortest_path(graph, start_node, end_node, weight='length')
    path_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in shortest_path_nodes]  # (lat, lon)
    return path_coords

# Placeholder function for the GAP model
def apply_gap_model(path_coords):
    # Placeholder function, add GAP model logic here if needed
    return path_coords

# Function to generate the final GPX with optimal path and existing route
def generate_path_gpx(start_point, end_point, bbox):
    graph = fetch_road_data_from_osm(bbox)
    optimal_path_coords = find_optimal_path_gap(start_point, end_point, graph)

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for lat, lon in optimal_path_coords:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon))

    return gpx

def gpx_to_json(gpx, team_name, trackerid, id_count):
    positions = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                id_count += 1
                json_point = {
                    "id": id_count,
                    "trackerID": trackerid,
                    "time": point.time.isoformat() + "Z",  # Convert to ISO 8601 format
                    "alt": point.elevation,
                    "lat": point.latitude,
                    "long": point.longitude
                    # Add other attributes as needed
                }
                positions.append(json_point)
    return id_count, positions

# Function to filter points based on start time
def filter_points_by_starttime(gpx, start_time_utc):
    for track in gpx.tracks:
        for segment in track.segments:
            segment.points = [point for point in segment.points if start_time_utc < point.time]
    return gpx

# Function to filter points based on end time
def filter_points_by_endtime(gpx, end_time_utc):
    for track in gpx.tracks:
        for segment in track.segments:
            segment.points = [point for point in segment.points if end_time_utc > point.time]
    return gpx

from datetime import timedelta

# Function to calculate travel time considering GAP adjustments
def calculate_travel_time(track_gpx, graph):
    updated_points = []

    for track in track_gpx.tracks:
        for segment in track.segments:
            for i in range(len(segment.points) - 1):
                point_i = segment.points[i]
                point_i1 = segment.points[i + 1]

                # Calculate grade between two points (example calculation)
                # Replace with your actual grade calculation based on elevation change
                grade = (point_i1.elevation - point_i.elevation) / point_i.distance_2d(point_i1)
                point_i.speed = point_i.distance_3d(point_i1)/point_i.time_difference(point_i1)

                # Calculate speed in m/s using GAP adjusted speed
                speed = SpdtoGAP(point_i.speed, grade)

                # Calculate time difference between two points based on adjusted speed
                time_diff = point_i1.time - point_i.time

                if time_diff.total_seconds() == 0:
                    continue

                # Find the nearest nodes on the road network to point_i and point_i1
                start_node = ox.distance.nearest_nodes(graph, point_i.longitude, point_i.latitude)
                end_node = ox.distance.nearest_nodes(graph, point_i1.longitude, point_i1.latitude)

                # Find the shortest path between the nodes
                try:
                    shortest_path_nodes = nx.shortest_path(graph, start_node, end_node, weight='length')
                except nx.NetworkXNoPath:
                    # If no path is found, append the direct path points and continue
                    updated_points.append(point_i)
                    updated_points.append(point_i1)
                    continue

                # Calculate total distance of the path
                path_length = nx.shortest_path_length(graph, start_node, end_node, weight='length')

                # Adjust speed along the path using GAP
                previous_time = point_i.time
                previous_point = Point(point_i.longitude, point_i.latitude)
                updated_points.append(point_i)

                for j in range(1, len(shortest_path_nodes) - 1):
                    node = shortest_path_nodes[j]
                    current_point = Point(graph.nodes[node]['x'], graph.nodes[node]['y'])

                    # Calculate distance and time to the current node
                    distance = previous_point.distance(current_point)
                    elapsed_time = timedelta(seconds=distance / speed)

                    # Create a new GPX point
                    gpx_point = gpxpy.gpx.GPXTrackPoint(current_point.y, current_point.x)
                    gpx_point.time = previous_time + elapsed_time
                    gpx_point.speed = speed  # Store speed in the GPX point for future use
                    updated_points.append(gpx_point)

                    previous_time = gpx_point.time
                    previous_point = current_point

                updated_points.append(point_i1)

    # Create a new GPX object to return
    updated_gpx = gpxpy.gpx.GPX()
    updated_track = gpxpy.gpx.GPXTrack()
    updated_gpx.tracks.append(updated_track)
    updated_segment = gpxpy.gpx.GPXTrackSegment()
    updated_track.segments.append(updated_segment)

    # Add all updated points to the GPX segment
    for point in updated_points:
        updated_segment.points.append(point)

    return updated_gpx



# Function to process GPX and generate final route prediction
async def main():
    end_point = gpxpy.gpx.GPXTrackPoint(latitude=-35.474519, longitude=148.2733177,
                                        elevation=393)  # Replace with actual endpoint
    id_count = 99999
    json_url = "https://live.anuinwardbound.com/api/2023/teams"
    response = requests.get(json_url)
    race_data = response.json()

    end_time_utc = datetime(2023, 10, 7, 8, 38) - timedelta(hours=11)
    droptimes = [
        datetime(2023, 10, 6, 20, 00),
        datetime(2023, 10, 6, 20, 45),
        datetime(2023, 10, 6, 21, 45),
        datetime(2023, 10, 6, 23, 30),
        datetime(2023, 10, 7, 0, 45),
        datetime(2023, 10, 7, 1, 15),
        datetime(2023, 10, 7, 4, 15)
    ]

    teams_data = []

    for team_data in race_data.get("teams", []):
        team_name = team_data.get('name')
        if len(team_data['tracker']['positions']) > 0:
            divno = int(re.search(r'\d+', team_name).group())

            start_time_utc = droptimes[divno - 1] - timedelta(hours=11)

            if datetime.now(timezone.utc).replace(tzinfo=None) < start_time_utc + timedelta(minutes=30):
                print("DIV " + str(divno) + " has not left yet")
                continue

            points_data = team_data['tracker']['positions']

            # Convert points_data to GPX asynchronously
            track_gpx = await points_to_gpx(points_data, team_name)

            # Apply filters
            track_gpx = filter_points_by_starttime(track_gpx, start_time_utc - timedelta(minutes=15))
            track_gpx = filter_points_by_endtime(track_gpx, end_time_utc)
            folder_path = os.path.join(os.getcwd(), "routes")
            file_path = os.path.join(folder_path, "0.gpx")
            with open(file_path, "r") as f:
                routegpx = gpxpy.parse(f)
            # Extract bbox from the track_gpx
            south = min(point.latitude for track in routegpx.tracks for segment in track.segments for point in segment.points)
            north = max(point.latitude for track in routegpx.tracks for segment in track.segments for point in segment.points)
            east = min(point.longitude for track in routegpx.tracks for segment in track.segments for point in segment.points)
            west = max(point.longitude for track in routegpx.tracks for segment in track.segments for point in segment.points)
            bbox = (north, south, east, west)
            graph=fetch_road_data_from_osm(bbox)
            # Calculate travel time and update track_gpx
            track_gpx = calculate_travel_time(track_gpx, graph)

            # Get the last point of the last segment of the last track in track_gpx
            current_point = track_gpx.tracks[-1].segments[-1].points[-1]

            # Generate optimal route GPX from current_point to end_point
            optimal_route_gpx = generate_path_gpx(current_point, end_point, graph)
            for track in optimal_route_gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        track_gpx.tracks[-1].segments[-1].points.append(point)
            # Convert GPX to JSON
            id_count, poslist = gpx_to_json(track_gpx, team_name, team_data['tracker']['id'], id_count)

            # Update team_data with modified positions
            team_data['tracker']['positions'] = poslist

            # Append modified team data
            teams_data.append(team_data)

    # Update race_data with modified teams data
    race_data['teams'] = teams_data

    # Save race predictions to JSON file
    with open('predictions.json', 'w') as json_file:
        json.dump(race_data, json_file)


# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())

