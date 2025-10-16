from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, gridplot
from bokeh.models import Slider, ColumnDataSource, CheckboxGroup, Select, Div, ColorBar, LinearColorMapper, GeoJSONDataSource
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.transform import linear_cmap
from shapely.geometry import mapping, shape
import pyproj
from pyproj import Proj, Transformer
from matplotlib.transforms import Affine2D
from shapely.affinity import affine_transform
from itertools import islice
import geopandas as gpd
import numpy as np
import pandas as pd
import ast
import json


# Petite fonction préliminaire

def str_to_list(s):
    return ast.literal_eval(s)


# Charger le fichier GeoJSON contenant les frontières des départements
with open("departements-geojson.json", "r") as f:
    geojson_data = json.load(f)

# Convertir les géométries GeoJSON en objets Shapely
geometries = [shape(feature['geometry']) for feature in geojson_data['features']]

# Création d'un GeoDataFrame à partir des géométries
geojson_data = gpd.GeoDataFrame(geometry=geometries)

scale_factor_x = 73.08
scale_factor_y = 111.111
offset_x = -20.276
offset_y = -5279.995

#scale_factor_x = 1
#scale_factor_y = 1
#offset_x = 0
#offset_y = 0

#transform = Affine2D().scale(73.08, 111.111).translate(offset_x, offset_y)

# Appliquer la transformation affine à la géométrie
#geojson_data['geometry'] = geojson_data['geometry'].apply(lambda geom: transform.transform(geom))
geojson_data['geometry'] = geojson_data['geometry'].apply(lambda geom: affine_transform(geom, [scale_factor_x, 0, 0, scale_factor_y, offset_x, offset_y]))

# Extraire les géométries et les convertir en dictionnaires GeoJSON
features = []
for index, row_ in geojson_data.iterrows():
    feature = {
        'type': 'Feature',
        'properties': {},
        'geometry': mapping(row_['geometry'])
    }
    features.append(feature)

# Créer un objet GeoJSON
geojson_data = {
    'type': 'FeatureCollection',
    'features': features
}


def print_premieres_lignes_geojson(geojson_data, nb_caracteres_max):
    for feature in geojson_data['features']:
        print("Geometry:")
        geometry_str = str(feature['geometry'])
        if len(geometry_str) > nb_caracteres_max:
            print(geometry_str[:nb_caracteres_max] + "...")
        else:
            print(geometry_str)
        print()

#print(geojson_data)
#debut_dictionnaire = dict(islice(geojson_data.items(), 2))
#for cle, valeur in debut_dictionnaire.items():
#    print(cle, ":", valeur)

#print_premieres_lignes_geojson(geojson_data, 200)

# Charger une nouvelle source de données GeoJSON avec les coordonnées transformées
geo_source = GeoJSONDataSource(geojson=json.dumps(geojson_data))

# Définition des limites de la carte.
x_left = 0
x_right = 300
y_top = 300
y_bottom = 0


def generation_plot(nombre_clusters, normalisation, rm_outliers):
    chemin_proportions = ".\Prop_clusters\Répartition_" + str(nombre_clusters) + "_clusters_06-09_" + "2018_norm=" + normalisation + "_rm_outliers=" + rm_outliers + ".csv"
    tableau_proportions_df = pd.read_csv(chemin_proportions)
    tableau_prop = tableau_proportions_df.iloc[:, 1:]
    tableau_prop_lst = tableau_prop.applymap(str_to_list)
    list_of_lists = tableau_prop_lst.values.tolist()
    if len(set(map(len, list_of_lists))) != 1:
        print("Les listes n'ont pas toutes la même longueur.")
    else:
        tableau_prop_lst = np.array(list_of_lists)
    k = tableau_prop_lst.shape[2]
    plots = []

    for l in range(k):
        carte_répartition = tableau_prop_lst[:, :, l] / tableau_prop_lst.sum(axis=2)
        mapper = LinearColorMapper(palette="Greys256", low=carte_répartition.min(), high=carte_répartition.max())
        p = figure(title='Répartition du cluster {}'.format(l), x_range=(0, tableau_prop_lst.shape[0]),
                   y_range=(0, tableau_prop_lst.shape[1]), width=480, height=400)
        #p.image(image=[carte_répartition], x=0, y=0, dw=tableau_prop_lst.shape[0], dh=tableau_prop_lst.shape[1],
        #        palette="Greys256")
        p.image(image=[carte_répartition], x=0, y=0, dw=tableau_prop_lst.shape[0], dh=tableau_prop_lst.shape[1],
                color_mapper = mapper)
        color_bar = ColorBar(color_mapper=mapper, width=8, location=(0,0))
        p.add_layout(color_bar, 'right')
        p.xaxis.axis_label = 'longitude'
        p.yaxis.axis_label = 'latitude'
        #p.patches('xs', 'ys', source=ColumnDataSource(df_geojson), line_color='red', line_width=0.5, fill_alpha=0)
        p.patches(xs='xs', ys='ys', 
          source=geo_source, 
          fill_color=None, 
          line_color='red', line_width=1)
        plots.append(p)

    plots_row = row(*plots)  # Convertir la liste de figures en une rangée de figures
    return (plots_row)

# Création d'une fonction pour mettre à jour la figure en fonction des paramètres choisis.

def update_plot(attr, old, new):
    rm_outliers = select_rm_outliers.value
    normalisation = select_normalisation.value
    nombre_clusters = nombre_clusters_slider.value
    new_plot = generation_plot(nombre_clusters, normalisation, rm_outliers)
    layout.children[3] = new_plot
    #div.text = file_html(new_plot, INLINE, "Plot")


# Création des widgets interactifs

select_rm_outliers = Select(title="Remove outliers", options=["yes", "no"], value="yes")
select_rm_outliers.on_change('value', update_plot)
select_normalisation = Select(title="Normalisation", options=["Divmax", "Standard"], value="Divmax")
select_normalisation.on_change('value', update_plot)
nombre_clusters_slider = Slider(title="Nombre de clusters", value=5, start=2, end=7, step=1)
nombre_clusters_slider.on_change('value', update_plot)

# Créer une figure initiale avec le premier paramètre
rm_outliers_ini = select_rm_outliers.value
normalisation_ini = select_normalisation.value
nombre_clusters_ini = nombre_clusters_slider.value
initial_plot = generation_plot(nombre_clusters_ini, normalisation_ini, rm_outliers_ini)

# Convertir la figure initiale en HTML
#initial_plot_html = file_html(initial_plot, INLINE, "Plot")

# Création d'un élément div pour afficher la figure
#div = Div(text=initial_plot_html, width=400, height=300)

# Créer la mise en page
global_title = Div(text="<h1>Répartition spatiale des clusters obtenus par l'algorithme K-Means</h1>")
#grid = gridplot([initial_plot])
mapper = linear_cmap(field_name='values', palette='Viridis256', low=0, high=1)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
layout = column(select_rm_outliers, select_normalisation, nombre_clusters_slider, initial_plot)
#final_layout = column(
#    row(layout, color_bar),
#    sizing_mode='stretch_both'  # Ajuste la taille du layout
#)

# Ajouter la mise en page à l'application Bokeh
curdoc().add_root(global_title)
curdoc().add_root(layout)

