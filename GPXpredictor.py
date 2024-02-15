import datetime
import os
import gpxpy.gpx
import pandas as pd
from numpy import average
from datetime import datetime, timedelta,timezone
import json
import pytz
import csv
import requests
import re
import xlsxwriter
def remove_numbers(input_string):
    return re.sub(r'\d', '', input_string)
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
def gpx_to_json(gpx, team_name):#we'll  need the uno reverse card to make it go backwards I porbably should have worked just in json but had this code already so just bodged it
    json_data = {
        "name": team_name,
        "positions": [],  # Initialize positions if not a guess
    }
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                json_point = {
                    "sogKmph": point.speed * 3.6 if point.speed is not None else None,  # Convert m/s to km/h
                    "txAt": point.time.isoformat() + "Z",  # Convert to ISO 8601 format
                    "gpsAtMillis": point.time.timestamp() * 1000,  # Convert to milliseconds
                    "altitude": point.elevation,
                    "latitude": point.latitude,
                    "longitude": point.longitude
                    # Add other attributes as needed
                }
                json_data["positions"].append(json_point)


    return json_data
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
                        if point.distance_3d(previous_point) > 0:
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
                    distance_meters = point.distance_3d(previous_point)

                    # Check if the distance is less than the maximum allowed gap
                    if distance_meters <= max_gap_distance_meters:
                        filtered_segment.points.append(point)

                previous_point = point

            if filtered_segment.points:
                filtered_track = gpxpy.gpx.GPXTrack()  # Create a new track for filtered data
                filtered_track.segments.append(filtered_segment)
                filtered_gpx.tracks.append(filtered_track)
    return filtered_gpx

def distance_filter_with_future_points(gpx, max_gap_distance_meters, n): # it's still pairwise but dtermines how far you look for the pair
    filtered_gpx = gpxpy.gpx.GPX()  # Create a new GPX object for filtered data
    for track in gpx.tracks:
        for segment in track.segments:
            filtered_segment = gpxpy.gpx.GPXTrackSegment()  # Create a new track segment for filtered data
            i = 0  # Index of the current point
            while i < len(segment.points):
                point = segment.points[i]
                future_point = segment.points[min(i+n,len(segment.points)-1)]
                if point.distance_3d(future_point)<max_gap_distance_meters:
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
def filter_points_by_time(gpx, start_time_utc):
    for track in gpx.tracks:
        for segment in track.segments:
            segment.points = [point for point in segment.points if start_time_utc < point.time]
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
def timeguess(teaminfo, routeinfo, teamname): #need to also add name info and drop time info to cull bus time from tracker data (and maybe finish time too)
    gapdf = pd.DataFrame(columns=['Time', "GAP"])
    gpx = teaminfo
    routeguess = gpxpy.gpx.GPX() #correctedroute
    endpoint = routeinfo.tracks[0].segments[-1].points[-1] #get the endpoint make sure first route has it
    eproutes = gpxpy.gpx.GPX()#make list of routes that go to endpoint as some tracks will merely be different ways between other points
    for track in routeinfo.tracks:
        trackep=track.segments[-1].points[-1]
        distance_to_endpoint = endpoint.distance_3d(trackep)
        if distance_to_endpoint < 300:#allow some buffer
            eproutes.tracks.append(track)
    for track in gpx.tracks:#make assumptions about intermediate points travelled to by chucking in route data between the points provided
        for segment in track.segments:
            for i in range(len(segment.points) - 1):
                point_i = segment.points[i]
                point_i1 = segment.points[i + 1]
                time = point_i.time_difference(point_i1)
                if time != 0:
                    locinfoi1= routeinfo.get_nearest_location(point_i1)  # get the point and track we closest to based on the second point (need to probably improve this algo)
                    locpointi1 = locinfoi1.point_no
                    loctracki1 = locinfoi1.track_no
                    locinfoi = routeinfo.tracks[loctracki1].get_nearest_location(point_i)
                    locpointi = locinfoi.point_no  # get the point and track we closest to based on the second point (need to probably improve this algo)
                    segmentinfo = gpxpy.gpx.GPX()
                    segmentinfo.tracks.append(routeinfo.tracks[loctracki1].clone())
                    if point_i.distance_3d(locinfoi.location) > 1000 or point_i1.distance_3d(locinfoi1.location) > 1000: # if corrected route more than 500m from one point don't correct
                        segmentinfo.tracks[0].segments[0].points = [point_i, point_i1]
                    elif locpointi1 >= locpointi:
                        segmentinfo.tracks[0].segments[0].points = [point_i]+routeinfo.tracks[loctracki1].segments[0].points[locpointi:locpointi1+1]+[point_i1]
                    else:
                        segmentinfo.tracks[0].segments[0].points =[point_i]+routeinfo.tracks[loctracki1].segments[0].points[locpointi:locpointi1+1:-1]+[point_i1]
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
    routeguess = filter_gpx(routeguess,1,20) # topspeed faster than a 2hr marathon
    routeguess = distance_filter_with_future_points(routeguess,2000,10)#try get rid of teh bus trip automagically done after the speed correction as that should remove most bus travel bits
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
    endpointinfo=routeinfo.get_nearest_location(endpoint) #get the point we closest to
    locpoint = endpointinfo.point_no #pointinfo
    locsegment = endpointinfo.segment_no
    loctrack =endpointinfo.track_no #trackchooser (may need to do some priority thing here)
    pathtofinish = gpxpy.gpx.GPX()
    pathtofinish.tracks.append(routeinfo.tracks[loctrack].clone())
    pathtofinish.tracks[0].segments[0].points = [endpoint]+routeinfo.tracks[loctrack].segments[locsegment].points[locpoint:]
    pathend=pathtofinish.tracks[0].segments[-1].points[-1]
    if pathend.distance_3d(endpoint) > 300:#if route doesn't go to endpoint repeat merging with a route that does
        endpointinfo = eproutes.get_nearest_location(pathend)  # get the point we closest to
        locpoint = endpointinfo.point_no# pointinfo
        locsegment = endpointinfo.segment_no
        loctrack = endpointinfo.track_no
        pathtofinish.tracks[0].segments[0].points += eproutes.tracks[loctrack].segments[locsegment].points[locpoint:]
    nettime = routeguess.get_duration()
    print(nettime)
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
    lasttime = routeguess.tracks[-1].segments[-1].points[-1].time
    n = 1
    while lasttime == None:
        if len(routeguess.tracks[-1].segments[-1])==0:
            routeguess.tracks[-1].segments.pop()
        if len(routeguess.tracks[-1])==0:
            routeguess.tracks.pop()
        lasttime = routeguess.tracks[-1].segments[-1].points[-1].time

    routeguess = infer_speed(routeguess)
    outgpx=routeguess.to_xml()
    folder_path = os.path.join(os.getcwd(), "teams")  # lets get a routes folder
    file_path = os.path.join(folder_path, f"{teamname}.gpx")
    with open(file_path,'w') as f:#outputs corrected/predictive gpx by team_name
        f.write(outgpx)

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
routegpx = gpxpy.gpx.GPX() # a store of all possible routes
folder_path=os.path.join(os.getcwd(),"routes")#lets get a routes folder
timeguesses={}
json_url = "https://yb.tl/API3/Race/inwardbound_hq2023/GetPositions?t=0"#change this to whatever the url is
response = requests.get(json_url)
race_data = response.json()
#this time filterinbg is just to test predictiveness
start_time_aedt = datetime(2022, 10, 30, 7, 8)  # need to make to drop time + scout time to remove idle time
end_time_aedt = datetime(2022, 10, 30, 8, 38)  # will be unnecessary as with cut to current time
start_time_utc = start_time_aedt - timedelta(hours=11)
end_time_utc = end_time_aedt - timedelta(hours=11)
droptimes=[datetime(2023, 10, 6, 20, 00),
datetime(2023, 10, 6, 20, 45),
datetime(2023, 10, 6, 21, 45),
datetime(2023, 10, 6, 23, 30),
datetime(2023, 10, 7, 0, 45),
datetime(2023, 10, 7, 1, 15),
datetime(2023, 10, 7, 4, 15)
]
teamtimedict={}
teams_data = []
teams = race_data.get("teams", [])
for team_data in teams:
    team_name = team_data.get('name')
    if "Driver" not in team_name and "Rescue" not in team_name and "Committee" not in team_name:
        # Extract the point data for this team
        routegpx = gpxpy.gpx.GPX()
        divno=get_divnumber(team_name)
        filename = str(divno)+".gpx"
        print(filename)
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            routegpx = gpxpy.parse(f)
        start_time_utc= droptimes[divno-1]- timedelta(hours=11)
        print(team_name)
        if datetime.now(timezone.utc).replace(tzinfo=None) < start_time_utc+timedelta(minutes=30):
            print("DIV " +str(divno)+ " has not left yet")
            continue
        if divno == 0  or "Division X" in team_name:
            print(team_name + "has no  divno")
            continue
        points_data = team_data.get('positions', [])
        track_gpx = points_to_gpx(points_data, team_name)
        track_gpx = filter_points_by_time(track_gpx, start_time_utc-timedelta(minutes=15))
        teamguess=timeguess(track_gpx,routegpx,team_name)
        teamtimedict[team_name] = teamguess[0]
        guess_json = gpx_to_json(teamguess[1],team_name)
        teams_data.append(guess_json)

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
with pd.ExcelWriter('racepredictions.xlsx', engine='xlsxwriter') as writer:
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

    # Apply the time format to columns 'B' and 'C'
    worksheet.set_column('B:W', None, time_format)
final_json = {
    "raceUrl": race_data.get("raceUrl"),  # Make whateever the race info is

    "teams": teams_data
}
with open('race_data.json', 'w') as json_file:
    json.dump(final_json, json_file)