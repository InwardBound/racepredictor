import datetime
import os
import pickle

import gpxpy.gpx
import gpxpy
import pandas as pd
from numpy import average
from datetime import datetime, timedelta,timezone
import json
import pytz
import csv
import requests
import re
import xlsxwriter
import asyncio
import aiohttp
import networkx as nx
import itertools
import random
from lxml import etree
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Point, Polygon, LineString


def remove_graph_points(graph, areas):
    # Load areas from JSON file
    nodes_to_remove = set()
    # Iterate through each area and mark nodes for removal
    for area in areas:
        if area['type'] == 'polygon':
            polygon = Polygon(area['coordinates'])
            for node in graph.nodes:
                point = Point(graph.nodes[node]['longitude'], graph.nodes[node]['latitude'])
                if polygon.contains(point):
                    nodes_to_remove.add(node)

        elif area['type'] == 'line':
            line = LineString(area['coordinates'])
            threshold = area.get('threshold', 0.01)
            for node in graph.nodes:
                point = Point(graph.nodes[node]['longitude'], graph.nodes[node]['latitude'])
                if point.distance(line) <= threshold:
                    nodes_to_remove.add(node)

    # Remove nodes and their associated edges
    graph.remove_nodes_from(nodes_to_remove)

    return graph
async def assign_unique_ids(routegpx):
    counter = itertools.count(start=1)
    all_points = [point for route in routegpx.tracks for segment in route.segments for point in segment.points]
    elevations = await fetch_elevations_batch_async(all_points)
    for point, id, elevation in zip(all_points, counter, elevations):
        if isinstance(point, gpxpy.gpx.GPXTrackPoint):
            point.name = str(id)
            point.elevation = elevation
    return routegpx

# Function to convert XML element node to a dictionary

def routegraph(route_data):
    G = nx.DiGraph()
    nodelist = []

    # Add nodes to the graph
    for track in route_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                point.time = None
                G.add_node(point, id=point.name)
                nodelist.append(point)

    # Add edges to the graph
    for i in range(len(nodelist)):
        point_i = nodelist[i]
        for search_radius in [10,20,50,100,200,500,1000]:
            edge_added=False
            for j in range(i + 1, len(nodelist)):
                point_j = nodelist[j]
                # Calculate distance between point_i and point_j
                distance = point_i.distance_2d(point_j)
                # Check if the distance exceeds the search radius
                if distance > search_radius:
                    continue
                point_k = nodelist[j-1]
                if point_i.distance_2d(point_k) < distance and point_k.distance_2d(point_j) < distance:#if earlier point is some midpoint skip shortcutting
                    continue
                # Calculate other metrics and add edges accordingly
                if distance == 0:
                    G.add_edge(point_i, point_j, weight=0)
                    G.add_edge(point_j, point_i, weight=0)
                else:
                    elevation_change = point_j.elevation - point_i.elevation
                    grade = elevation_change / point_i.distance_2d(point_j)
                    time_forward = distance / GAPtoSPD(2, min(max(grade,-.3),.3))
                    time_backward = distance / GAPtoSPD(2, min(max(-grade,-.3),.3))
                    G.add_edge(point_i, point_j, weight=time_forward)
                    G.add_edge(point_j, point_i, weight=time_backward)
                    edge_added=True
            if edge_added:
                break
    return G
def string_to_trackpoint(s):
    # Remove unnecessary characters and split the string
    s = s.strip('[trkpt:]')
    parts = s.split('@')
    # Extract latitude, longitude, elevation (and potentially other attributes)
    coords = parts[0].split(',')
    latitude = float(coords[0])
    longitude = float(coords[1])
    elevation = float(parts[1]) if len(parts) > 1 else None
    # Create and return GPXTrackPoint object
    trackpoint = gpxpy.gpx.GPXTrackPoint(latitude=latitude, longitude=longitude, elevation=elevation)
    return trackpoint

def find_optimal_route(start_point_id, end_point_id, graph):
    # Lookup points by their IDs in the graph
    start_point = None
    end_point = None
    for node in graph.nodes:
        if str(node) == str(start_point_id):
            start_point = node
        elif str(node) == str(end_point_id):
            end_point = node
    if start_point is None or end_point is None:
        raise ValueError("Start or end point IDs not found in the graph.")

    # Find the optimal path using shortest path algorithm
    # Create a GPX object to store the optimal route
    gpx_optimal = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx_optimal.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    try:
        optimal_path = nx.shortest_path(graph, source=start_point, target=end_point, weight='weight')
        # Add points of the optimal route to the GPX segment
        for point in optimal_path:
            gpx_segment.points.append(string_to_trackpoint(graph.nodes[point]['label']))
    except:# iff no path found just assume travelled directly between those two points
        gpx_segment.points.append(string_to_trackpoint(graph.nodes[start_point]['label']))
        gpx_segment.points.append(string_to_trackpoint(graph.nodes[end_point]['label']))

    return gpx_segment.points


def remove_points_in_area(graph, min_latitude, max_latitude, min_longitude, max_longitude):
    nodes_to_remove = [
        node for node in graph.nodes
        if min_latitude <= node.latitude <= max_latitude and min_longitude <= node.longitude <= max_longitude
    ]
    # Remove identified nodes and their associated edges
    graph.remove_nodes_from(nodes_to_remove)
    return graph
def get_nearest_location(point, graph):
    # Initialize variables to store nearest information
    nearest_node = None
    min_distance = float('inf')

    # Iterate through all nodes in the graph
    for node in graph.nodes:
        # Calculate distance between the given point and the current node
        distance = point.distance_2d(node)

        # Update nearest node if the current node is closer than previously found nodes
        if distance < min_distance:
            nearest_node = node
            min_distance = distance

    return nearest_node
def remove_numbers(input_string):
    return re.sub(r'\d', '', input_string)
def SpdtoGAP(speed, x):
    multiplier = 15.377 * x**5 - 41.537 * x**4 - 8.1948 * x**3 + 18.516 * x**2 + 3.122 * x + 1
    return (speed*multiplier)
def GAPtoSPD(GAP, x):
    divisor = 15.377 * x**5 - 41.537 * x**4 - 8.1948 * x**3 + 18.516 * x**2 + 3.122 * x + 1
    return (GAP/divisor)


# Asynchronously fetch elevation data in batches with retry mechanism and different User-Agent headers
async def fetch_elevations_batch_async(points):
    url = "https://api.opentopodata.org/v1/aster30m?locations="
    batched_points = [points[i:i + 100] for i in range(0, len(points), 100)]
    all_elevations = []
    async with aiohttp.ClientSession() as session:
        for batch in batched_points:
            if hasattr(points[0], 'latitude'):
                batch_url = url + "|".join(f"{point.latitude},{point.longitude}" for point in batch)
            else:
                batch_url = url + "|".join(f"{lat},{lon}" for lat, lon in batch)
            print('fetching',batch_url)
            retry_attempts = 5
            for attempt in range(retry_attempts):
                try:
                    async with session.get(batch_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if "results" in data and len(data["results"]) > 0:
                                elevations = [result["elevation"] for result in data["results"]]
                                all_elevations.extend(elevations)
                                break
                            else:
                                print("No elevation data found in API response")
                                break
                        elif response.status == 429:
                            print(f"Rate limited by API, retrying in {2 ** attempt} seconds...")
                            await asyncio.sleep(2 ** attempt)
                        else:
                            print(f"Error fetching elevation data: HTTP {response.status}")
                            break
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


def filter_gpx(gpx, min_speed_kmph, max_speed_kmph): #removes teleporting maybe bussing too
    filtered_gpx = gpxpy.gpx.GPX()  # Create a new GPX object for filtered data
    for track in gpx.tracks:
        for segment in track.segments:
            filtered_segment = gpxpy.gpx.GPXTrackSegment()  # Create a new track segment for filtered data
            previous_point = None
            for point in segment.points:
                if previous_point is not None:
                    # Check if required data for speed calculation is available
                    if point.time and previous_point.time:
                        if (point.time - previous_point.time).total_seconds() > 0:
                            # Calculate speed between two points (in km/h)
                            speed_kmph = point.distance_3d(previous_point) / (point.time - previous_point.time).total_seconds() * 3.6  # Convert m/s to km/h
                            if min_speed_kmph <= speed_kmph <= max_speed_kmph:
                                filtered_segment.points.append(previous_point)
                previous_point = point
            if filtered_segment.points:
                filtered_track = gpxpy.gpx.GPXTrack()  # Create a new track for filtered data
                filtered_track.segments.append(filtered_segment)
                filtered_gpx.tracks.append(filtered_track)

    return filtered_gpx
def distance_filter(gpx, max_gap_distance_meters):#this basic pairwise approach ain't good enough
    filtered_gpx = gpxpy.gpx.GPX()  # Create a new GPX object for filtered data
    for track in gpx.tracks:
        for segment in track.segments:
            filtered_segment = gpxpy.gpx.GPXTrackSegment()  # Create a new track segment for filtered data
            previous_point = None

            for point in segment.points:
                if previous_point is not None:
                    # Calculate distance between current point and previous point
                    distance_meters = point.distance_2d(previous_point)

                    # Check if the distance is less than the maximum allowed gap
                    if distance_meters <= max_gap_distance_meters:
                        filtered_segment.points.append(point)

                previous_point = point

            if filtered_segment.points:
                filtered_track = gpxpy.gpx.GPXTrack()  # Create a new track for filtered data
                filtered_track.segments.append(filtered_segment)
                filtered_gpx.tracks.append(filtered_track)
    return filtered_gpx
def gpx_to_json(gpx, team_name, trackerid,id_count):#we'll need the uno reverse card to make it go backwards I porbably should have worked just in json but had this code already so just bodged it
    positions=[]
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                id_count += 1
                json_point = {
                "id": id_count,
                "trackerID": trackerid,
                "time": point.time.isoformat() + "Z", # Convert to ISO 8601 format
                "alt": point.elevation,
                "lat": point.latitude,
                "long": point.longitude
                # Add other attributes as needed
                }
                positions.append(json_point)
    return id_count, positions

def distance_filter_with_future_points(gpx, max_gap_distance_meters, n): # it's still pairwise but dtermines how far you look for the pair
    filtered_gpx = gpxpy.gpx.GPX()  # Create a new GPX object for filtered data
    for track in gpx.tracks:
        for segment in track.segments:
            filtered_segment = gpxpy.gpx.GPXTrackSegment()  # Create a new track segment for filtered data
            i = 0  # Index of the current point
            while i < len(segment.points):
                point = segment.points[i]
                future_point = segment.points[min(i+n,len(segment.points)-1)]
                if point.distance_2d(future_point)<max_gap_distance_meters:
                    filtered_segment.points.append(point)
                    i += 1  # Move to the next point
                else:
                    # Skip points with no future points within the specified distance
                    i += n
            if filtered_segment.points:
                filtered_track = gpxpy.gpx.GPXTrack()  # Create a new track for filtered data
                filtered_track.segments.append(filtered_segment)
                filtered_gpx.tracks.append(filtered_track)
    return filtered_gpx


# Function to filter GPX points within a specific time range
def filter_points_by_starttime(gpx, start_time_utc):
    for track in gpx.tracks:
        for segment in track.segments:
            segment.points = [point for point in segment.points if start_time_utc < point.time]
    return gpx
def filter_points_by_endtime(gpx, end_time_utc):
    for track in gpx.tracks:
        for segment in track.segments:
            segment.points = [point for point in segment.points if end_time_utc > point.time]
    return gpx
def infer_speed(gpx):
    for track in gpx.tracks:
        for segment in track.segments:
            previous_point = None
            for point in segment.points:
                if previous_point is not None:
                    previous_point.speed = previous_point.speed_between(point)
                previous_point = point
    return gpx
def timeguess(teaminfo, routeinfo, graph,endpoint,teamname):
    gapdf = pd.DataFrame(columns=['Time', "GAP",'Duration'])
    gpx = teaminfo
    gpx = filter_gpx(gpx, 0, 30)  # Top speed faster than a 2-hour marathon
    routeguess = gpxpy.gpx.GPX()  # Corrected route GPX object
    for track in gpx.tracks:
        for segment in track.segments:
            for w in range(len(segment.points)-1):
                point_i = segment.points[w]
                point_i1 = segment.points[w+1]
                time = point_i.time_difference(point_i1)
                if time != 0:
                    locinfoi1 = routeinfo.get_nearest_location(point_i1)  # Get closest point and track based on the second point
                    p_i1 = locinfoi1.point_no
                    s_i1 = locinfoi1.segment_no
                    t_i1 = locinfoi1.track_no
                    i1close= routeinfo.tracks[t_i1].segments[s_i1].points[p_i1]
                    locinfoi = routeinfo.get_nearest_location(point_i)
                    p_i = locinfoi.point_no
                    s_i = locinfoi.segment_no
                    t_i = locinfoi.track_no
                    iclose = routeinfo.tracks[t_i].segments[s_i].points[p_i]
                    segmentinfo = gpxpy.gpx.GPX()  # GPX object to store segment info
                    segmentinfo.tracks.append(gpxpy.gpx.GPXTrack())
                    segmentinfo.tracks[0].segments.append(gpxpy.gpx.GPXTrackSegment())
                    # If corrected route is more than 500m from one point, don't correct
                    if iclose.name == i1close.name\
                            or  point_i1.distance_2d(locinfoi1.location) > point_i1.distance_2d(point_i)\
                            or  point_i.distance_2d(locinfoi.location) > point_i1.distance_2d(point_i):
                        segmentinfo.tracks[0].segments[0].points = [point_i, point_i1]
                        speed = segmentinfo.length_3d() / time  # Need to adjust for GAP
                    elif point_i.distance_2d(locinfoi.location) > 500 or \
                            point_i1.distance_2d(locinfoi1.location) > 500:
                        [point_i] + find_optimal_route(str(iclose.name),str(i1close.name), graph) + [point_i1]
                    else:
                        segmentinfo.tracks[0].segments[0].points =  find_optimal_route(str(iclose.name),str(i1close.name), graph)
                        segmentinfo.tracks[0].segments[0].points[0].time = point_i.time
                        segmentinfo.tracks[0].segments[0].points[-1].time = point_i1.time
                        speed = segmentinfo.length_3d() / time  # Need to adjust for GAP
                    if speed >7:#assume dem error
                        speed = segmentinfo.length_2d() / time
                else:
                    speed = 0
                if speed != 0:  # No point if same point
                    for t in segmentinfo.tracks:
                        for s in t.segments:
                            previous_time = None
                            previous_point = None
                            for p in s.points:
                                if previous_time is not None and previous_point is not None:
                                    # Calculate distance between previous point and current point
                                    distance = previous_point.distance_3d(p)
                                    # Calculate elapsed time between previous point and current point
                                    elapsed_time = timedelta(seconds=distance / speed)
                                    # Set the time of the current point based on the elapsed time
                                    p.time = previous_time + elapsed_time
                                # Update previous_time and previous_point for next iteration
                                previous_time = p.time
                                previous_point = p

                    if len(routeguess.tracks) == 0:
                        routeguess.tracks.append(segmentinfo.tracks[0])
                    else:
                        routeguess.tracks[0].segments[0].points.extend(segmentinfo.tracks[0].segments[0].points)
    # Remove likely errors from GPX
    routeguess = filter_gpx(routeguess, 0, 20)  # Top speed faster than a 2-hour marathon
    routeguess = distance_filter_with_future_points(routeguess, 2000, 5)  # Try to remove bus trips automatically
    timecount=0
    for track in routeguess.tracks:
        for segment in track.segments:
            for i in range(len(segment.points) - 1):
                point_i = segment.points[i]
                point_i1 = segment.points[i + 1]
                time = point_i.time_difference(point_i1)
                timecount += time
                vert = point_i1.elevation - point_i.elevation
                dist = point_i.distance_2d(point_i1)
                speed=0
                if time>0:
                    speed = point_i.distance_3d(point_i1)/time
                if speed>7 and time>0:#if really fast assume dem error
                    speed=dist/time
                if dist != 0 and speed < 7:
                    grade = vert / dist
                    GAPspeed = SpdtoGAP(speed, max(min(grade,.3),-.3))#limits grade to .3 as gap starts go weird when really steep
                    inputdf = [time, GAPspeed, timecount]
                    gapdf.loc[len(gapdf)] = inputdf
                if point_i1.distance_2d(endpoint) < 300:
                    currentpointinfo = routeguess.get_nearest_location(point_i1)
                    locpoint = currentpointinfo.point_no
                    locsegment = currentpointinfo.segment_no
                    loctrack = currentpointinfo.track_no
                    finishedroute = routeguess
                    finishedroute.tracks = finishedroute.tracks[:loctrack + 1]
                    finishedroute.tracks[-1].segments = finishedroute.tracks[-1].segments[:locsegment + 1]
                    finishedroute.tracks[-1].segments[-1].points = finishedroute.tracks[-1].segments[-1].points[:locpoint + 1]
                    lasttime = point_i1.time
                    folderpath = os.path.join(os.getcwd(), "teams")
                    filepath = os.path.join(folderpath, teamname + "(finished).gpx")
                    with open(filepath, "w") as f:
                        f.write(finishedroute.to_xml())
                    return [lasttime + timedelta(hours=11),finishedroute]  # Return the complete predicted route as GPX
    currentpoint = routeguess.tracks[-1].segments[-1].points[-1]
    recentdf = gapdf[gapdf['Duration'] > max(gapdf['Duration']) - 7200]#only consider most recent 2hrs of running
    NetGAP = average(recentdf['GAP'], weights=recentdf['Time'])
    currentpointinfo = routeinfo.get_nearest_location(currentpoint)  # Get the closest point to the current point
    p_i = currentpointinfo.point_no
    s_i = currentpointinfo.segment_no
    t_i = currentpointinfo.track_no
    cp = routeinfo.tracks[t_i].segments[s_i].points[p_i]
    ep =routeinfo.tracks[-1].segments[-1].points[-1]
    pathtofinish = gpxpy.gpx.GPX()
    pathtofinish.tracks.append(gpxpy.gpx.GPXTrack())
    pathtofinish.tracks[0].segments.append(gpxpy.gpx.GPXTrackSegment())
    pathtofinish.tracks[0].segments[0].points = [currentpoint] + find_optimal_route(cp.name,ep.name, graph)
    nettime = routeguess.get_duration()
    for track in pathtofinish.tracks:
        for segment in track.segments:
            for point_i, point_i1 in zip(segment.points, segment.points[1:]):
                vert = point_i1.elevation - point_i.elevation
                dist = point_i.distance_2d(point_i1)
                if dist != 0:
                    grade = vert / dist
                    speed = GAPtoSPD(NetGAP, max(min(grade,.3),-.3))
                    timeguess = dist / speed
                    nettime += timeguess
                    point_i1.time = point_i.time + timedelta(seconds=timeguess)
                else:
                    point_i1.time = point_i.time + timedelta(seconds=1)#add a section of waiting just not duplicate shit
    for track in pathtofinish.tracks:
        routeguess.tracks.append(track)

    lasttime = routeguess.tracks[-1].segments[-1].points[-1].time
    n = 1

    while lasttime == None:
        if len(routeguess.tracks[-1].segments[-1]) == 0:
            routeguess.tracks[-1].segments.pop()

        if len(routeguess.tracks[-1]) == 0:
            routeguess.tracks.pop()

        lasttime = routeguess.tracks[-1].segments[-1].points[-1].time

    routeguess = infer_speed(routeguess)
    folderpath = os.path.join(os.getcwd(), "teams")
    filepath = os.path.join(folderpath, teamname+".gpx")
    with open(filepath, "w") as f:
        f.write(routeguess.to_xml())
    return [lasttime + timedelta(hours=11),routeguess] #convert back to aedt


def get_divnumber(input_string):
    # Use regular expression to find the number at the end of the string
    match = re.search(r'\d+$', input_string)

    if match:
        # If a number is found, return it as an integer
        return int(match.group())
    else:
        # If no number is found, return 0
        return 0


async def main():
    graphcache='graph.gexf'
    gpxcache = 'routes.gpx'
    endpoint = gpxpy.gpx.GPXTrackPoint(latitude=-35.474536, longitude=148.275841,elevation=393)
    #endpoint = gpxpy.gpx.GPXTrackPoint(latitude=-35.70982, longitude=150.250498,elevation=2) 2022 ep
    folder_path = os.path.join(os.getcwd(), "routes")
    file_path = os.path.join(folder_path, "0.gpx")
    with open(file_path, "r") as f:
        routegpx = gpxpy.parse(f)
    if os.path.exists(gpxcache):
        with open(gpxcache, "r") as f:
            routegpx = gpxpy.parse(f)
        if not os.path.exists(graphcache):
            route_graph = routegraph(routegpx)
            nx.write_gexf(route_graph, graphcache)
    else:
        routegpx.tracks.append(gpxpy.gpx.GPXTrack())
        routegpx.tracks[-1].segments.append(gpxpy.gpx.GPXTrackSegment())
        routegpx.tracks[-1].segments[-1].points.append(endpoint)
        routegpx=await assign_unique_ids(routegpx)
        with open(gpxcache, "w") as f:
            f.write(routegpx.to_xml())
        route_graph = routegraph(routegpx)
        nx.write_gexf(route_graph, graphcache)
    idcount=99999
    # Load race data from API (assuming this section is already defined in your script)
    json_url = "https://live.anuinwardbound.com/api/2023/teams"
    response = requests.get(json_url)
    race_data = response.json()
    # Define drop times and other required variables
    end_time_utc = datetime(2023, 10, 7, 8, 17)- timedelta(hours=11)
    teamtimedict = {}
    teams_data = []
    route_graph = nx.read_gexf(graphcache)
    # Iterate over teams in race data
    for team_data in race_data.get("teams", []):
        team_name = team_data.get('name')
        if len(team_data['tracker']['positions'])>0 and team_data['dnfAt'] is None:
            # Extract the point data for this team
            divno = get_divnumber(team_name)
            folder_path = os.path.join(os.getcwd(), "exclusions")
            file_path = os.path.join(folder_path, str(divno)+".json")
            div_route_graph = route_graph
            if os.path.exists(file_path):
                print(file_path)
                with open(file_path, 'r') as file:
                    areas_data = json.load(file)
                div_route_graph = remove_graph_points(routegraph,areas_data)
            points_data = team_data['tracker']['positions']
            tracker_id = team_data['tracker']['positions'][0]['trackerID']
            # Convert points_data to GPX asynchronously
            track_gpx = await points_to_gpx(points_data, team_name)
            # Apply filters
            track_gpx = filter_points_by_endtime(track_gpx, end_time_utc)
            # Perform time estimation
            teamguess = timeguess(track_gpx, routegpx,div_route_graph,endpoint,team_name)
            teamtimedict[team_name] = teamguess[0]
            # Convert GPX to JSON
            idcount, poslist = gpx_to_json(teamguess[1], team_name, tracker_id, idcount)
            team_data['tracker']['positions'] = poslist
            # Append modified team data
            teams_data.append(team_data)

    # Update race_data with modified teams data
    race_data['teams'] = teams_data

    # Save race predictions to JSON file
    with open('predictions.json', 'w') as json_file:
        json.dump(race_data, json_file)

    # Perform further data analysis and export to Excel
    df = pd.DataFrame(list(teamtimedict.items()), columns=['Name', 'Time'])

    # Extract the 'Team' names (remove all digits)
    df['Team'] = df['Name'].apply(lambda x: remove_numbers(x))
    # Extract the 'Number' from the 'Name' column
    df['Division'] = df['Name'].str.extract(r'(\d+)').fillna(1).astype(int)
    # Group by 'Team' and aggregate 'Time' values into a list
    grouped = df.groupby('Team')['Time'].agg(list).reset_index()
    df_pivot = df.pivot(index='Division', columns='Team', values='Time')
    # Reset the index to make 'Number' a column
    df_pivot.reset_index(inplace=True)
    print(df_pivot)

    # Export to Excel
    with pd.ExcelWriter('racepredictions(2hrtimewindow).xlsx', engine='xlsxwriter') as writer:
        df_pivot.to_excel(writer, sheet_name='racepredictions', index=False)

        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['racepredictions']

        # Define a custom number format for the 'Division' column (assuming it's in the first column)
        integer_format = workbook.add_format({'num_format': '0'})

        # Apply the integer format to the 'Division' column
        worksheet.set_column('A:A', None, integer_format)

        # Define a custom time format for other columns
        time_format = workbook.add_format({'num_format': 'h:mm:ss AM/PM'})

        # Apply the time format to columns 'B' and beyond (adjust 'W' as per your data)
        worksheet.set_column('B:W', None, time_format)


# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())