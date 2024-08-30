This is a flask app for predicting the eventual IB finish times.
It requires 5 things to run:
    1. The 'slug' of the race this should be the year
    2. The latitude, longitude and altitude of the endpoint of the race 
    3. The start and finish/cut-off time of the race so it knows when to run 
    4. All the possible routes teams can travel by so the model can convert this into a graph where it can find optimal routes, for this to best work, avoid overlapping paths as otherwise it willl be larger than neccessary, and do not omit any possible route, this can be created using gpx.studio or similar
    5. Any div-specific exclusion zones as defined by a geojson shape saved in the exclusions folder then the folder with the race slug and saved as the division name, make the zone if anything slightly too small as it only as to disconnect any routes going through for it not assume teams wilkl travel through there. These can be made at geojson.io
