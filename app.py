from flask import Flask, send_file, jsonify, url_for
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
from GPXpredictor import main  # Import the main function
import os
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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
    print( f"Running GPXpredictor with year={year}, latitude={latitude}, longitude={longitude}, elevation={elevation}...")
    await main(year, latitude, longitude, elevation)
    print(f"GPXpredictor execution for year {year} completed.")


def schedule_predictions_for_2024():
    current_year = "2024"

    if current_year in endpoints:
        latitude, longitude, elevation = endpoints[current_year]
    else:
        print(f"No endpoint found for year {current_year}, skipping...")
        return

    start_time = datetime(2024, 10, 11, 20, 0, 0, tzinfo=ZoneInfo('Australia/Sydney'))
    end_time = start_time + timedelta(hours=24)
    print(start_time)
    scheduler.add_job(
        lambda: asyncio.run(run_gpxpredictor(current_year, latitude, longitude, elevation)),
        'interval',
        minutes=4,
        start_date=start_time,
        end_date=end_time
    )

    print(f"Scheduled predictions for year {current_year} from {start_time} to {end_time}")


@app.route('/<year>', methods=['GET'])
def get_predictions(year):
    predictions_file = get_predictions_file_path(year)

    if year == "2024":
        if os.path.exists(predictions_file):
            return send_file(predictions_file, mimetype='application/json')
        else:
            return jsonify({"error": f"Predictions file for year {year} not found"}), 404
    else:
        if os.path.exists(predictions_file):
            return send_file(predictions_file, mimetype='application/json')
        else:
            return jsonify({"error": f"Predictions file for year {year} not found"}), 404


@app.route('/graphs/<raceslug>', methods=['GET'])
def serve_graph(raceslug):
    graphs_path = os.path.join(os.getcwd(), 'graphs', raceslug)

    if not os.path.exists(graphs_path):
        return jsonify({"error": f"No graph directory found for race slug {raceslug}"}), 404

    # List all HTML files in the race slug directory
    html_files = [f for f in os.listdir(graphs_path) if f.endswith('.html')]

    # Check if the current time is past the start time (example: 2024-10-10 20:00:00)
    start_time = datetime(2024, 10, 11, 19, 30, 0, tzinfo=ZoneInfo('Australia/Sydney'))
    current_time = datetime.now(ZoneInfo('Australia/Sydney'))

    if current_time < start_time and raceslug == str(start_time.year):
        return jsonify({"error": "Graphs not yet available"}), 403

    # Generate HTML response with hyperlinks to the HTML files
    links_html = "<h1>Available Graphs</h1><ul>"
    for html_file in html_files:
        file_url = url_for('get_graph_file', raceslug=raceslug, filename=html_file)
        links_html += f'<li><a href="{file_url}">{html_file}</a></li>'
    links_html += "</ul>"

    return links_html


@app.route('/graphs/<raceslug>/<filename>', methods=['GET'])
def get_graph_file(raceslug, filename):
    graphs_path = os.path.join(os.getcwd(), 'graphs', raceslug, filename)
    start_time = datetime(2024, 10, 11, 19, 30, 0, tzinfo=ZoneInfo('Australia/Sydney'))
    current_time = datetime.now(ZoneInfo('Australia/Sydney'))
    if current_time < start_time and raceslug == str(start_time.year):
        return jsonify({"error": "Graphs not yet available"}), 403

    if not os.path.exists(graphs_path):
        return jsonify({"error": f"File {filename} not found in race slug {raceslug}"}), 404

    # Serve the HTML file with caching headers
    response = send_file(graphs_path, mimetype='text/html')
    response.headers['Cache-Control'] = 'public, max-age=3333'  # Cache for an hourish
    return response
endpoints_file_path = os.path.join(os.getcwd(), 'endpoints.txt')
endpoints = load_endpoints(endpoints_file_path)

scheduler = BackgroundScheduler()

schedule_predictions_for_2024()

scheduler.start()

if __name__ == '__main__':


    port = int(os.getenv('PORT', 8000))

    try:
        app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
