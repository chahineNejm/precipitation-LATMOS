
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure, gridplot
from bokeh.models import Slider, ColumnDataSource, CheckboxGroup, Select, Div, ColorBar, LinearColorMapper, GeoJSONDataSource, DataTable, TableColumn
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.transform import linear_cmap
from bokeh.server.server import Server
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
import requests


# Petite fonction préliminaire qui permet de convertir un liste stockée sous forme de chaîne de caractères en une liste.

def str_to_list(s):
    return ast.literal_eval(s)


# Charger le fichier GeoJSON contenant les frontières des départements
with open("departements-geojson.json", "r") as f:
    geojson_data = json.load(f)

# Charger le fichier JSON contenant les cours d'eau
with open("rivers.json", "r") as g:
    geojson_data_hydro = json.load(g)

#geojson_data_hydro = dict(geojson_data_hydro)
#geojson_data_hydro = gpd.read_file("réseau_hydro.geojson")

with open("forets_idf.json", "r") as h:
    geojson_data_forest = json.load(h)

print(type(geojson_data))
print(type(geojson_data_hydro))
#print(geojson_data)
#print(geojson_data_hydro)
# Convertir les géométries GeoJSON en objets Shapely
geometries = [shape(feature['geometry']) for feature in geojson_data['features']]
geometries_hydro = [shape(feature['geo_shape']['geometry']) for feature in geojson_data_hydro]
geometries_forest = [shape(feature['geo_shape']['geometry']) for feature in geojson_data_forest]

# Création d'un GeoDataFrame à partir des géométries
geojson_data = gpd.GeoDataFrame(geometry=geometries)
geojson_data_hydro = gpd.GeoDataFrame(geometry=geometries_hydro)
geojson_data_forest = gpd.GeoDataFrame(geometry=geometries_forest)


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
geojson_data_hydro['geometry'] = geojson_data_hydro['geometry'].apply(lambda geom: affine_transform(geom, [scale_factor_x, 0, 0, scale_factor_y, offset_x, offset_y]))
geojson_data_forest['geometry'] = geojson_data_forest['geometry'].apply(lambda geom: affine_transform(geom, [scale_factor_x, 0, 0, scale_factor_y, offset_x, offset_y]))

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

features_hydro = []
for index, row_ in geojson_data_hydro.iterrows():
    feature = {
        'type': 'Feature',
        'properties': {},
        'geometry': mapping(row_['geometry'])
    }
    features_hydro.append(feature)

# Créer un objet GeoJSON
geojson_data_hydro = {
    'type': 'FeatureCollection',
    'features': features_hydro
}


features_forest = []
for index, row_ in geojson_data_forest.iterrows():
    feature = {
        'type': 'Feature',
        'properties': {},
        'geometry': mapping(row_['geometry'])
    }
    features_forest.append(feature)

# Créer un objet GeoJSON
geojson_data_forest = {
    'type': 'FeatureCollection',
    'features': features_forest
}


def print_premieres_lignes_geojson(geojson_data, nb_caracteres_max, stop=10):
    i = 0
    for feature in geojson_data['features']:
        if (i<10):
            print("Geometry:")
            geometry_str = str(feature['geometry'])
            if len(geometry_str) > nb_caracteres_max:
                print(geometry_str[:nb_caracteres_max] + "...")
            else:
                print(geometry_str)
            print()
            i=i+1

#print(geojson_data)
#debut_dictionnaire = dict(islice(geojson_data.items(), 2))
#for cle, valeur in debut_dictionnaire.items():
#    print(cle, ":", valeur)

# print("Premières lignes du fichier des départements : ")
# print_premieres_lignes_geojson(geojson_data, 100)
# print("Premières lignes du fichier des cours d'eau : ")
# print_premieres_lignes_geojson(geojson_data_hydro, 100)
# print("Premières lignes du fichier des forêts : ")
# print_premieres_lignes_geojson(geojson_data_forest, 100)

# Charger de nouvelles sources de données GeoJSON avec les coordonnées transformées
geo_source = GeoJSONDataSource(geojson=json.dumps(geojson_data))
geo_source_hydro = GeoJSONDataSource(geojson=json.dumps(geojson_data_hydro))
geo_source_forest = GeoJSONDataSource(geojson=json.dumps(geojson_data_forest))

# Définition des limites de la carte.
x_left = 0
x_right = 300
y_top = 300
y_bottom = 0


## Cette fonction génère les différents graphes à afficher, en fonction des paramètres qui lui sont passés en arguments.
def generation_plot(nombre_clusters, normalisation, rm_outliers, selected_options, nombre_plots=10):
    chemin_proportions = ".\Prop_clusters\Répartition_" + str(nombre_clusters) + "_clusters_06-09_" + "2018_norm=" + normalisation + "_rm_outliers=" + rm_outliers + ".csv"
    tableau_proportions_df = pd.read_csv(chemin_proportions)
    centres_clusters = tableau_proportions_df.iloc[:5,1:12]
    tableau_prop = tableau_proportions_df.iloc[5:, 1:]
    tableau_prop_lst = tableau_prop.applymap(str_to_list)
    list_of_lists = tableau_prop_lst.values.tolist()
    if len(set(map(len, list_of_lists))) != 1:
        print("Les listes n'ont pas toutes la même longueur.")
    else:
        tableau_prop_lst = np.array(list_of_lists)
    #k = tableau_prop_lst.shape[2]
    k = nombre_clusters
    plots = []
    text_centres_clusters = []

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
        if ('Limites des départements' in selected_options):
            p.patches(xs='xs', ys='ys', 
            source=geo_source, 
            fill_color=None, 
            line_color='red', alpha = 1, line_width=1)
            #print("Limites des départements ajoutées !")
        else : 
            p.patches(xs='xs', ys='ys', 
            source=geo_source, 
            fill_color=None, 
            line_color='red', alpha = 0, line_width=1)
            #print("Limites des départements supprimées.")
        if ("Cours d'eau" in selected_options):
            p.patches(xs='xs', ys='ys', 
            source=geo_source_hydro, 
            fill_color=None, 
            line_color='blue', alpha=1, line_width=1)
            #print("Cours d'eau ajoutés !")
        else :
            p.patches(xs='xs', ys='ys', 
            source=geo_source_hydro, 
            fill_color=None, 
            line_color='blue', alpha=0, line_width=1)
            #print("Cours d'eau supprimés.")
        if ("Limites des massifs forestiers" in selected_options):
            p.patches(xs='xs', ys='ys', 
            source=geo_source_forest, 
            fill_color=None, 
            line_color='green', alpha=1, line_width=0.2)
            #print("Forêts ajoutées !")
        else : 
            p.patches(xs='xs', ys='ys', 
            source=geo_source_forest, 
            fill_color=None, 
            line_color='green', alpha=0, line_width=0.2)
            #print("Limites des massifs forestiers supprimées.")
        plots.append(p)
        
        # Ajout des centres des clusters en-dessous des cartes géographiques : 
        centre_cluster = centres_clusters.iloc[:,[0,l+1]]
        columns = []
        for column_name in centre_cluster.columns:
            columns.append(TableColumn(field=column_name, title=column_name))
            #columns.append(Div(text=column_name))
        source = ColumnDataSource(centre_cluster)
        data_table = DataTable(source=source, columns=columns, width=400, height=200)
        #text_widget = Div(text= centres_clusters.iloc[:,[0,l+1]])
        explanation_text = "<h2>Centre du cluster {}:</h2>".format(l)
        text_centres_clusters.append(column(Div(text=explanation_text), data_table))

    for l in range (k,nombre_plots):
        #carte_répartition = pd.DataFrame(0, index=range(300), columns=range(300))
        carte_répartition = np.zeros((300, 300))
        mapper = LinearColorMapper(palette="Greys256", low=0, high=255)
        p = figure(title='Répartition du cluster {}'.format(l), x_range=(0, tableau_prop_lst.shape[0]),
                   y_range=(0, tableau_prop_lst.shape[1]), width=480, height=400)
        p.image(image=[carte_répartition], x=0, y=0, dw=tableau_prop_lst.shape[0], dh=tableau_prop_lst.shape[1],
                color_mapper = mapper)
        #print(p)
        p.patches(xs='xs', ys='ys', 
            source=geo_source, 
            fill_color=None, 
            line_color='red', alpha = 0, line_width=1)
        
        p.patches(xs='xs', ys='ys', 
            source=geo_source_hydro, 
            fill_color=None, 
            line_color='blue', alpha=0, line_width=1)
        
        p.patches(xs='xs', ys='ys', 
            source=geo_source_forest, 
            fill_color=None, 
            line_color='green', alpha=0, line_width=0.2)
        
        plots.append(p)
        
        centre_cluster = centres_clusters.iloc[:,[0,1]]
        centre_cluster.loc[:, "1"] = "0"
        #print(centre_cluster)
        columns = []
        for column_name in centre_cluster.columns:
            columns.append(TableColumn(field=column_name, title=column_name))
        source = ColumnDataSource(centre_cluster)
        data_table = DataTable(source=source, columns=columns, width=400, height=200)
        #text_widget = Div(text= centres_clusters.iloc[:,[0,l+1]])
        explanation_text = "<h2>Centre du cluster {}:</h2>".format(l)
        #text_centres_clusters.append(column(data_table))
        text_centres_clusters.append(column(Div(text=explanation_text), data_table))
        
    #plots_row = row(*plots)  # Convertir la liste de figures en une rangée de figures
    #texts_row = row(*text_centres_clusters)
    return (plots, text_centres_clusters)

# Création d'une fonction pour mettre à jour la figure en fonction des paramètres choisis.

def update_plot(attr, old, new):
    print("Mise à jour de l'interface")
    global plots
    global texts
    global layout
    global global_title
    global document
    rm_outliers = select_rm_outliers.value
    normalisation = select_normalisation.value
    nombre_clusters = nombre_clusters_slider.value
    selected_options = [checkbox_labels[i] for i in checkbox_group.active]
    print(selected_options)
    new_plots, new_texts = generation_plot(nombre_clusters, normalisation, rm_outliers, selected_options)
    
    # Mettre à jour les graphes existants avec les nouveaux graphes
    for i, new_plot in enumerate(new_plots):
        if i < len(plots):
            # print(plots[i].renderers)
            # print(new_plot.renderers)
            # for j in range (0,len(plots[i].renderers)):
            #     plots[i].renderers[j].data_source.data.update(new_plot.renderers[j].data_source.data)
            plots[i].renderers[0].data_source.data.update(new_plot.renderers[0].data_source.data)
            # plots[i].renderers[1].data_source.geojson = new_plot.renderers[1].data_source.geojson
            # plots[i].renderers[2].data_source.geojson = new_plot.renderers[2].data_source.geojson
            # plots[i].renderers[3].data_source.geojson = new_plot.renderers[3].data_source.geojson
            plots[i].renderers[1] = new_plot.renderers[1]
            plots[i].renderers[2] = new_plot.renderers[2]
            plots[i].renderers[3] = new_plot.renderers[3]
        else:
            # Si new_plots est plus long, ajouter de nouveaux graphes à ploSts
            plots.append(new_plot)
    
    for i, new_text in enumerate(new_texts):
        if i < len(plots):
            texts[i].children[1].source.data.update(new_text.children[1].source.data)
        else:
            # Si new_plots est plus long, ajouter de nouveaux graphes à plots
            texts.append(new_text)



# Création des widgets interactifs

select_rm_outliers = Select(title="Remove outliers", options=["yes", "no"], value="yes")
select_rm_outliers.on_change('value', update_plot)
select_normalisation = Select(title="Normalisation", options=["Divmax", "Standard"], value="Standard")
select_normalisation.on_change('value', update_plot)
nombre_clusters_slider = Slider(title="Nombre de clusters", value=10, start=2, end=10, step=1)
nombre_clusters_slider.on_change('value', update_plot)
checkbox_labels = ["Limites des départements", "Cours d'eau", "Limites des massifs forestiers"]  # Liste des noms des cours d'eau
checkbox_group = CheckboxGroup(labels=checkbox_labels, active=[0, 1, 2])
#checkbox_group = CheckboxGroup(labels=checkbox_labels)
checkbox_group.on_change('active', update_plot)

# Créer une figure initiale avec le premier paramètre
rm_outliers_ini = select_rm_outliers.value
normalisation_ini = select_normalisation.value
nombre_clusters_ini = nombre_clusters_slider.value
selected_options = [checkbox_labels[i] for i in checkbox_group.active]
plots, texts = generation_plot(nombre_clusters_ini, normalisation_ini, rm_outliers_ini, selected_options)

# for i, (plot, text) in enumerate(zip(plots, texts)):
#     plot.name = f"plot_{i}"  # Attribuer un nom unique à chaque plot
#     text.name = f"text_{i}"  # Attribuer un nom unique à chaque text

grid = gridplot([[plot, text] for plot, text in zip(plots, texts)])

# Créer la mise en page
global_title = Div(text="<h1>Répartition spatiale des clusters obtenus par l'algorithme K-Means</h1>")
#grid = gridplot([initial_plot])
mapper = linear_cmap(field_name='values', palette='Viridis256', low=0, high=1)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8, location=(0,0))
layout = column(select_rm_outliers, select_normalisation, nombre_clusters_slider, checkbox_group, grid)
#final_layout = column(
#    row(layout, color_bar),
#    sizing_mode='stretch_both'  # Ajuste la taille du layout
#)

# Ajouter la mise en page à l'application Bokeh
document = curdoc()
document.add_root(global_title)
document.add_root(layout)

# Obtenir la configuration actuelle de la session Bokeh
config = Server({}).session_config

# Modifier la valeur de min_delay
config.min_delay = 5000  # Temps en millisecondes

# Appliquer les modifications à la session Bokeh
Server({}).session_config = config
