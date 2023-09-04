import datetime
import os
import gpxpy.gpx
import pandas as pd
from numpy import average
from datetime import datetime, timedelta
import json
import pytz
import csv
import requests
def SpdtoGAP(speed, grade):
    multiplier = 14.636*grade*grade+2.8609*grade+1
    return (speed*multiplier)
def GAPtoSPD(GAP, grade):
    divisor = 14.636 * grade * grade + 2.8609 * grade + 1
    return (GAP/divisor)
def points_to_gpx(points_data, track_name):
    gpx = gpxpy.gpx.GPX()
    # Create a track with the specified name
    track = gpxpy.gpx.GPXTrack(name=track_name)
    gpx.tracks.append(track)
    # Create a segment for the track
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for point in reversed(points_data):
        lat = point['latitude']
        lon = point['longitude']
        elevation = point['altitude']
        time = datetime.strptime(point['gpsAt'], '%Y-%m-%dT%H:%M:%SZ')
        gpx_point = gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=elevation, time=time)
        segment.points.append(gpx_point)
    return gpx
def filter_gpx(gpx, min_speed_kmph, max_speed_kmph): #removes teleporting maybe bussing too
    filtered_gpx = gpxpy.gpx.GPX()  # Create a new GPX object for filtered data
    for track in gpx.tracks:
        for segment in track.segments:
            filtered_segment = gpxpy.gpx.GPXTrackSegment()  # Create a new track segment for filtered data
            previous_point = None
            for point in segment.points:
                if previous_point is None:
                    filtered_segment.points.append(point)
                    previous_point = point
                    continue

                # Check if required data for speed calculation is available
                if point.time and previous_point.time:
                    if point.distance_3d(previous_point) > 0:
                        # Calculate speed between two points (in km/h)
                        speed_kmph = point.distance_3d(previous_point) / (point.time - previous_point.time).total_seconds() * 3.6  # Convert m/s to km/h
                        if min_speed_kmph <= speed_kmph <= max_speed_kmph:
                            filtered_segment.points.append(point)
                previous_point = point
            if filtered_segment.points:
                filtered_track = gpxpy.gpx.GPXTrack()  # Create a new track for filtered data
                filtered_track.segments.append(filtered_segment)
                filtered_gpx.tracks.append(filtered_track)

    return filtered_gpx

    return filtered_gpx
routegpx = gpxpy.gpx.GPX() # a store of all possible routes
folder_path=os.path.join(os.getcwd(),"routes")#lets get a routes folder
timeguesses={}
for file in os.listdir(folder_path):
    if file.endswith(".gpx"):
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            gpx_file = gpxpy.parse(f)
        # add each track in the file to the main GPX object
        for track in gpx_file.tracks:
            routegpx.tracks.append(track)
# Function to filter GPX points within a specific time range
def filter_points_by_time(gpx, start_time_utc, end_time_utc):
    for track in gpx.tracks:
        for segment in track.segments:
            segment.points = [point for point in segment.points if start_time_utc < point.time < end_time_utc]
    return gpx
def timeguess(teaminfo, routeinfo, teamname): #need to also add name info and drop time info to cull bus time from tracker data (and maybe finish time too)
    gapdf = pd.DataFrame(columns=['Time', "GAP"])
    gpx = teaminfo
    routeguess = gpxpy.gpx.GPX() #correctedroute
    for track in gpx.tracks:#make assumptions about intermediate points travelled to by chucking in route data between the points provided
        for segment in track.segments:
            for i in range(len(segment.points) - 1):
                point_i = segment.points[i]
                point_i1 = segment.points[i + 1]
                time = point_i.time_difference(point_i1)
                if time != 0:
                    locinfoi1= routegpx.get_nearest_location(point_i1)  # get the point and track we closest to based on the second point (need to probably improve this algo)
                    locpointi1 = locinfoi1.point_no
                    loctracki1 = locinfoi1.track_no
                    locinfoi = routegpx.tracks[loctracki1].get_nearest_location(point_i)
                    locpointi = locinfoi.point_no  # get the point and track we closest to based on the second point (need to probably improve this algo)
                    segmentinfo = gpxpy.gpx.GPX()
                    segmentinfo.tracks.append(routegpx.tracks[loctracki1].clone())
                    if point_i.distance_3d(locinfoi.location) > 500 or point_i1.distance_3d(locinfoi1.location) > 500: # if corrected route more than 500m from one point don't correct
                        segmentinfo.tracks[0].segments[0].points = [point_i, point_i1]
                    elif locpointi1 >= locpointi:
                        segmentinfo.tracks[0].segments[0].points = [point_i]+routegpx.tracks[loctracki1].segments[0].points[locpointi:locpointi1+1]+[point_i1]
                    else:
                        segmentinfo.tracks[0].segments[0].points =[point_i]+routegpx.tracks[loctracki1].segments[0].points[locpointi:locpointi1+1:-1]+[point_i1]
                    speed = segmentinfo.length_3d()/time #need to gap adjust
                else:
                    speed = 0
                if speed!=0:#pointless if same point
                    for t in segmentinfo.tracks:
                        for s in t.segments:
                            previous_time = None
                            previous_point = None
                            for p in s.points:
                                if previous_time is not None and previous_point is not None:
                                    # Calculate distance between previous point and current point
                                    distance = previous_point.distance_3d(p)
                                    # Calculate elapsed time between previous point and current point
                                    elapsed_time = timedelta(seconds=distance/speed)
                                    # Set the time of the current point based on the elapsed time
                                    p.time = previous_time + elapsed_time
                                # Update previous_time and previous_point for next iteration
                                previous_time = p.time
                                previous_point = p
                    if len(routeguess.tracks) == 0:
                        routeguess.tracks.append(segmentinfo.tracks[0])
                    else:
                        routeguess.tracks[0].segments[0].points.extend(segmentinfo.tracks[0].segments[0].points)
   #remove likely errors for gpx
    routeguess = filter_gpx(routeguess,0.1,21) # topspeed faster than a 2hr marathon
    for track in routeguess.tracks:
        for segment in track.segments:
            for i in range(len(segment.points)-1):
                point_i=segment.points[i]
                point_i1 = segment.points[i+1]
                speed = point_i.speed_between(point_i1)
                time = point_i.time_difference(point_i1)
                vert = point_i1.elevation - point_i.elevation
                dist = point_i.distance_3d(point_i1)
                if dist !=0:
                    grade=vert/dist
                    GAPspeed = SpdtoGAP(speed, grade)
                    inputdf = [time,GAPspeed]
                    gapdf.loc[len(gapdf)]=inputdf
                if i == len(segment.points)-2:
                    endpoint = point_i1 #where we end and guess what route we are on from
    NetGAP=average(gapdf['GAP'], weights = gapdf['Time'])
    endpointinfo=routegpx.get_nearest_location(endpoint) #get the point we closest to
    locpoint = endpointinfo.point_no #pointinfo
    loctrack =endpointinfo.track_no #trackchooser (may need to do some priority thing here)
    pathtofinish = gpxpy.gpx.GPX()
    pathtofinish.tracks.append(routegpx.tracks[loctrack].clone())
    pathtofinish.tracks[0].segments[0].points = [endpoint]+routegpx.tracks[loctrack].segments[0].points[locpoint:]
    nettime = routeguess.get_duration()
    for track in pathtofinish.tracks:
        for segment in track.segments:
            previous_time = None
            previous_point = None
            for point_i, point_i1 in zip(segment.points, segment.points[1:]):
                vert = point_i1.elevation - point_i.elevation
                dist = point_i.distance_2d(point_i1)
                if dist != 0:
                    grade = vert / dist
                    speed = GAPtoSPD(NetGAP, grade)
                    timeguess = dist / speed
                    nettime += timeguess
                    point_i1.time = point_i.time + timedelta(seconds=timeguess)
                previous_time = point_i.time
                previous_point = point_i
    for track in pathtofinish.tracks:
       routeguess.tracks.append(track)
    outgpx=routeguess.to_xml()
    folder_path = os.path.join(os.getcwd(), "teams")  # lets get a routes folder
    file_path = os.path.join(folder_path, f"{teamname}.gpx")
    with open(file_path,'w') as f:#outputs corrected/predictive gpx by team_name
        f.write(outgpx)
    return routeguess.tracks[-1].segments[-1].points[-1].time + timedelta(hours=11) #convert back to aedt

json_url = "https://yb.tl/API3/Race/inwardbound2022/GetPositions?t=0"
response = requests.get(json_url)
race_data = response.json()
teamtimedict={}
teams = race_data.get("teams", [])
aedt = pytz.timezone('Australia/Sydney')
start_time_aedt = datetime(2022, 10, 30, 6, 30)  # need to make to drop time + scout time to remove idle time
end_time_aedt = datetime(2022, 10, 30, 8, 38)  # will be unnecessary as with cut to current time
start_time_utc = start_time_aedt - timedelta(hours=11)
end_time_utc = end_time_aedt - timedelta(hours=11)
for team_data in teams:
    team_name = team_data.get('name')
    if "DRIVER" not in team_name:
        # Extract the point data for this team
        points_data = team_data.get('positions', [])
        track_gpx = points_to_gpx(points_data, team_name)
        track_gpx = filter_points_by_time(track_gpx, start_time_utc, end_time_utc)
        teamtimeguess=timeguess(track_gpx,routegpx,team_name)
        teamtimedict[team_name] = teamtimeguess
with open('2022resultsguess.csv', 'w') as predictions:
    writer = csv.writer(predictions)
    writer.writerow(teamtimedict.keys())
    writer.writerow(teamtimedict.values())