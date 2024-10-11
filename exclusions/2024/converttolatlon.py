import os
import geojson
from pyproj import Transformer

# Define the transformer to convert from EPSG:7855 to EPSG:4326 (lat, lon)
transformer = Transformer.from_crs("EPSG:7855", "EPSG:4326", always_xy=True)

# Function to transform coordinates recursively, handling MultiPolygons and other geometries
def transform_coordinates(coordinates):
    if isinstance(coordinates[0], (list, tuple)):  # Nested list for MultiPolygons or Polygons
        return [transform_coordinates(coord) for coord in coordinates]
    else:  # Single point (for Point geometries)
        lon, lat = transformer.transform(coordinates[0], coordinates[1])
        return [lon, lat]

# Process all GeoJSON FeatureCollections in the current folder
def process_geojson_files_in_current_folder():
    current_folder = os.getcwd()  # Get the current working directory

    # Loop through all files in the current folder
    for filename in os.listdir(current_folder):
        if filename.endswith(".geojson"):
            input_path = os.path.join(current_folder, filename)
            output_path = os.path.join(current_folder, f"transformed_{filename}")

            # Load the GeoJSON file
            with open(input_path, "r") as f:
                data = geojson.load(f)

            # Ensure the file is a FeatureCollection
            if data.get("type") == "FeatureCollection":
                # Check for CRS to confirm it's EPSG:7855
                if data.get("crs", {}).get("properties", {}).get("name") == "urn:ogc:def:crs:EPSG::7855":
                    # Iterate over each feature in the collection
                    for feature in data['features']:
                        geometry = feature['geometry']
                        if geometry['coordinates']:  # Only transform if coordinates are present
                            geometry['coordinates'] = transform_coordinates(geometry['coordinates'])

                    # Save the transformed GeoJSON to the same folder with a 'transformed_' prefix
                    with open(output_path, "w") as f:
                        geojson.dump(data, f)

                    print(f"Transformed: {filename} -> {output_path}")
                else:
                    print(f"Skipped (CRS not EPSG:7855): {filename}")
            else:
                print(f"Skipped (not a FeatureCollection): {filename}")

# Run the function to process all GeoJSON FeatureCollections in the current folder
process_geojson_files_in_current_folder()

print("All FeatureCollection files in the current folder have been transformed!")
