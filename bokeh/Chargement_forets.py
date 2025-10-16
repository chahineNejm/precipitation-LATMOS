## Ce fichier permet de télécherger les limites des massifs forestiers d'Île-de-France et de les stocker dans un 
## fichier json.


import osmnx as ox
import geopandas as gpd
import json
from shapely.geometry import mapping

# Configurer les paramètres d'OSMnx
ox.config(use_cache=True, log_console=True)

# Définir la région d'intérêt : Île-de-France
place_name = 'Île-de-France, France'

# Requête pour obtenir les massifs forestiers (landuse=forest)
tags = {'landuse': 'forest'}
gdf = ox.geometries_from_place(place_name, tags)

# Filtrer les massifs forestiers
massifs_forestiers = gdf[gdf['landuse'] == 'forest']

# Préparer la structure des données pour le format spécifié
features = []
for idx, row in massifs_forestiers.iterrows():
    if 'geometry' in row and row['geometry'] is not None:
        centroid = row['geometry'].centroid
        geo_json_feature = {
            "geo_point_2d": {
                "lon": centroid.x,
                "lat": centroid.y
            },
            "geo_shape": {
                "type": "Feature",
                "geometry": mapping(row['geometry'])
            }
        }
        features.append(geo_json_feature)

# Sauvegarder les données dans un fichier JSON
output_file = 'massifs_forestiers_ile_de_france_format_specifique.json'
with open(output_file, 'w') as f:
    json.dump(features, f, indent=2)

print(f'Données exportées dans le fichier {output_file}')