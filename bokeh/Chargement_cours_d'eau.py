
## Fichier de chargement des données concernant les cours d'eau

import json
from overpass import API

# Configuration de la zone géographique (bounding box)
bbox = (47, -0.5, 51, 4.5)  # Exemple de bounding box à Paris

# Initialisation de l'API Overpass
api = API()

# Requête pour récupérer les ways "waterway"="river" dans la bounding box
query = f"""
way["waterway"="river"]{bbox};
out geom;
"""

# Récupération des données
response = api.get(query)
data = response['features']

# Liste pour stocker les cours d'eau
rivers = []

# Parcourir les éléments récupérés
for feature in data:
    if 'geometry' in feature and feature['geometry']['type'] == 'LineString':
        coordinates = feature['geometry']['coordinates']
        
        # Vérifier que les coordonnées ne sont pas vides
        if coordinates:
            # Créer la structure JSON spécifiée
            river = {
                "geo_point_2d": {
                    "lon": coordinates[0][0],
                    "lat": coordinates[0][1]
                },
                "geo_shape": {
                    "type": "Feature",
                    "geometry": {
                        "coordinates": coordinates,
                        "type": "LineString"
                    }
                }
            }
            rivers.append(river)

# Écrire les données dans un fichier JSON
with open('rivers.json', 'w') as f:
    json.dump(rivers, f, indent=2)

print("Fichier rivers.json créé avec succès.")