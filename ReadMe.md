This is a flask app for predicting the eventual IB finish times.
It requires 5 things to run:
    1. The 'slug' of the race this should be the year in the env on fly.
    2. The latitude, longitude and altitude of the endpoint of the race set in endpoints.txt and the lat lon and alt of drop points in to csv file name [year].csv in dropoints folder.
    3. The start and finish/cut-off time of the race so it knows when to run in app.py.
    4. All the possible routes teams can travel by so the model can convert this into a graph where it can find optimal routes, for this to best work, avoid overlapping paths as otherwise it willl be larger than neccessary, and do not omit any possible route, this can be created using gpx.studio or similar, this as year.gpx in the course gpx folder.
    5. Any div-specific exclusion zones as defined by a geojson shape saved in the exclusions folder then the folder with the race slug and with the division numbers in them, any global ones must contain the word global intheir file name, if anything slightly too small as it only as to disconnect any routes going through for it not assume teams wilkl travel through there. These can be made at geojson.io
