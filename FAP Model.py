import gpxpy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def SpdtoGAP(speed, x):
    multiplier = 15.377 * x ** 5 - 41.537 * x ** 4 - 8.1948 * x ** 3 + 18.516 * x ** 2 + 3.122 * x + 1
    return (speed * multiplier)
# Function to parse GPX file
def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            points.extend(segment.points)
    return points
# Load GPX files from the 'finished' directory
gpx_dir = 'finished'  # Replace with your directory
gpx_files = [os.path.join(gpx_dir, file) for file in os.listdir(gpx_dir) if file.endswith('.gpx')]
gpx_points_list = [parse_gpx(file) for file in gpx_files]

times = []
GAPratio = []
distance = []
for gpx_points in gpx_points_list:
    GAPs = []
    elapsed_time = 0
    distance_covered = 0
    gapdf = pd.DataFrame(columns=['Time', 'GAP'])
    for i in range(len(gpx_points)-31)[::30]:
        time_50 = 0
        minigapdf = pd.DataFrame(columns=['Time', 'GAP'])
        for p in range(i,i+30):
            point = gpx_points[p]
            next_point = gpx_points[p + 1]
            vert = next_point.elevation - point.elevation
            dist2d = next_point.distance_2d(point)
            dist3d = next_point.distance_3d(point)
            time = next_point.time_difference(point)
            elapsed_time += time
            time_50 += time
            if dist2d > 0.0 and time > 0.0:
                speed = dist3d / time
                if speed > 7:
                    dist3d = dist2d
                GAP = SpdtoGAP(speed, min(max(vert / dist2d,.3),-.3))
                distance_covered += dist3d
                inputdf = [time, GAP]
                minigapdf.loc[len(minigapdf)] = inputdf
        gap_50 = np.average(minigapdf['GAP'], weights=minigapdf['Time'])
        gapdf.loc[len(gapdf)] = [time_50,gap_50]
        GAPs.append(gap_50)
        times.append(elapsed_time)
        distance.append(distance_covered)
    NetGAP = np.average(gapdf['GAP'], weights=gapdf['Time'])
    GAPs = [i / NetGAP for i in GAPs]
    GAPratio.extend(GAPs)

# Plotting GAPratio against times and distances
plt.figure(figsize=(14, 7))

# Plotting GAPratio vs Time
plt.subplot(1, 2, 1)
plt.plot(times, GAPratio, label='GAP Ratio')
plt.xlabel('Time (s)')
plt.ylabel('GAP Ratio')
plt.title('GAP Ratio over Time')
plt.legend()

# Plotting GAPratio vs Distance
plt.subplot(1, 2, 2)
plt.plot(distance, GAPratio, label='GAP Ratio')
plt.xlabel('Distance (m)')
plt.ylabel('GAP Ratio')
plt.title('GAP Ratio over Distance')
plt.legend()

plt.tight_layout()
plt.show()


# Fitting polynomial models to the data
def fit_polynomial_model(x, y, degree):
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(np.array(x).reshape(-1, 1))
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    mse = mean_squared_error(y, y_poly_pred)
    return model, mse, y_poly_pred


degrees = [2, 3, 4]
models = {}

for degree in degrees:
    model, mse, y_poly_pred = fit_polynomial_model(times, GAPratio, degree)
    models[degree] = (model, mse, y_poly_pred)
    plt.plot(times, y_poly_pred, label=f'Degree {degree} (MSE={mse:.2f})')

plt.scatter(times, GAPratio, color='blue', s=10, label='GAP Ratio')
plt.xlabel('Time (s)')
plt.ylabel('GAP Ratio')
plt.title('Polynomial Fit to GAP Ratio over Time')
plt.legend()
plt.show()

# Export the data to a spreadsheet
output_df = pd.DataFrame({'Time (s)': times, 'Distance (m)': distance, 'GAP Ratio': GAPratio})
output_file_path = 'GAPratio_data.xlsx'
output_df.to_excel(output_file_path, index=False)

print(f"Data exported to {output_file_path}")
