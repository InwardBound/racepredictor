from flask import Flask, send_file, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
from GPXpredictor import main  # Import the main function
import gpxpy
import os
import json
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

def get_predictions_file_path(year):
    return os.path.join(os.getcwd(), f'predictions_{year}.json')

def get_latest_prediction_time(predictions_file):
    with open(predictions_file, 'r') as json_file:
        race_data = json.load(json_file)
        latest_time = None
        for team_data in race_data.get("teams", []):
            eta = team_data.get("ETA")
            if eta:
                eta_datetime = datetime.fromisoformat(eta.replace("Z", "+00:00"))
                if not latest_time or eta_datetime > latest_time:
                    latest_time = eta_datetime
        return latest_time

def load_endpoints(file_path):
    endpoints = {}
    with open(file_path, 'r') as file:
        for line in file:
            year, coords = line.strip().split(' = ')
            coords = coords.strip('()')
            latitude, longitude, elevation = map(float, coords.split(','))
            endpoints[year] = (latitude, longitude, elevation)
    return endpoints

async def run_gpxpredictor(year, latitude, longitude, elevation):
    print(f"Running GPXpredictor with year={year}, latitude={latitude}, longitude={longitude}, elevation={elevation}...")
    await main(year, latitude, longitude, elevation)
    print(f"GPXpredictor execution for year {year} completed.")

@app.route('/api/predictions/<year>', methods=['GET'])
def get_predictions(year):
    predictions_file = get_predictions_file_path(year)
    if os.path.exists(predictions_file):
        return send_file(predictions_file, mimetype='application/json')
    else:
        return jsonify({"error": f"Predictions file for year {year} not found"}), 404


if __name__ == '__main__':
    # Load the endpoints from the file
    endpoints_file_path = os.path.join(os.getcwd(), 'endpoints.txt')
    endpoints = load_endpoints(endpoints_file_path)

    # Initialize the scheduler
    scheduler = BackgroundScheduler()

    # Process all GPX files in the `coursegpx` folder
    coursegpx_folder = os.path.join(os.getcwd(), 'coursegpx')
    for file_name in os.listdir(coursegpx_folder):
        if file_name.endswith('.gpx'):
            year = file_name.split('.')[0]  # Extract year from file name
            print(f"Processing year {year}...")

            if year in endpoints:
                latitude, longitude, elevation = endpoints[year]
            else:
                print(f"No endpoint found for year {year}, skipping...")
                continue

            # Run GPXpredictor for the year
            asyncio.run(run_gpxpredictor(year, latitude, longitude, elevation))

            # Check if the latest prediction time is more than an hour after the current time
            predictions_file = get_predictions_file_path(year)
            if os.path.exists(predictions_file):
                latest_time = get_latest_prediction_time(predictions_file)
                if latest_time and latest_time > datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=1):
                    print(f"Scheduling updates for year {year}...")
                    # Schedule updates for this year every 10 minutes
                    scheduler.add_job(lambda: asyncio.run(run_gpxpredictor(year, latitude, longitude, elevation)),
                                      'interval', minutes=10)
                else:
                    print(f"No updates scheduled for year {year}; latest prediction time is within the next hour.")
            else:
                print(f"No predictions file found for year {year}, so no updates are scheduled.")

    # Start the scheduler
    scheduler.start()

    try:
        # Start the Flask app
        app.run(debug=True, use_reloader=False)  # use_reloader=False to prevent APScheduler from running twice
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
