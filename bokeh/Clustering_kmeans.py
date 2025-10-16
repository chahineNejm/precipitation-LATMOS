
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import io
import time
import math
import os
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer


### IMPORTANT : Ce fichier de code a été mis au point avant que la construction de la base SQL des événements de pluie
### soit réalisée. Les événements de pluie sont donc ici exraits de dossiers zip. Quelques lignes de code devraient
### néanmoins permettre de charger directement les données de la base SQL.


path = 'C:/Users/Trist/Documents/Cours 2A - CS/Semestre 8/Projet S8/features_01012018.csv'
chemin_dossier_zip = "./precipitations/features_01_2018.zip"
#data_complète = pd.read_csv(path)
#data = data_complète.iloc[:,8:]

## Cette fonction permet de charger des données stockées dans un dossier zippé.
def chargement_données_zippées(chemin_dossier_zip):
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    df = pd.read_pickle(chemin_dossier_zip)
    df['end_time_absolute'] = df['start_time_absolute'] + df['duration'] # should be removed in the future (newly generated .pkl files will already have this column)
    return(df)


## Petite fonction qui divise chaque colonne par sa valeur maximale.
def max_scaling(X):
    return X / X.max(axis=0)

## Fonction qui enlève les outliers, c'est-à-dire les valeurs trop éloignées de la moyenne.
def remove_outliers(data, features):
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    print(Q1, Q3)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0 * IQR
    upper_bound = Q3 + 0 * IQR
    return data[(data[features] >= lower_bound) & (data[features] <= upper_bound)].dropna()

## Cette fonction enlève aussi les outliers, mais en détectant les outliers après nrmalisation standard.
def remove_outliers_zscore(data, features):
    z_scores = (data[features] - data[features].mean()) / data[features].std()
    return data[(z_scores.abs() < 3).all(axis=1)]

## Cette fonction permet d'arrondir le nombre number pour ne garder que significant_figures chiffres significatifs.
def round_to_significant_figures(number, significant_figures):
    if number == 0:
        return 0
    exposant = significant_figures - 1 - math.floor(math.log10(abs(number)))
    return (round(number*10**exposant)/(10**exposant))
    #return round(number, significant_figures - int(math.floor(math.log10(abs(number)))) - 1)


## On définit une fonction de clustering K-Means qui prétraite les données en enlevant ou non les outliers
# et en adaptant le mode de normalisation selon le paramètre du même nom qui ui est donné en entrée.

def clustering_KMeans(data_complète, k, normalisation="Standard", rm_outliers='no'):
    FEATURES = ['duration', 'max_intensity', 'mean_intensity', 'variance', 'percentage_null']
    means = data_complète[FEATURES].mean()
    stds = data_complète[FEATURES].std()
    maxima = data_complète[FEATURES].max(axis=0)
    print(maxima)
    data_complète_2 = data_complète
    if (rm_outliers == "yes"):
        data_complète_2 = remove_outliers_zscore(data_complète, FEATURES)
         # Vérifier si des données sont disponibles après la suppression des valeurs aberrantes
        if (data_complète_2.shape[0] == 0):
            print("Toutes les données ont été supprimées lors du processus de suppression des valeurs aberrantes.")
            return None
    print("Durée des événements avant normalisation : ")
    print(data_complète_2['duration'])
    if (normalisation == "Standard"):
        pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('clustering', KMeans(n_clusters=k, random_state=42))  
        # L'argument random_state est fixé afin d'avoir des résultats reproductibles. 
        # La valeur 42 est choisie par défaut.
        ])
    elif (normalisation == "Divmax"):
        pipeline = Pipeline([
        ('scaling', FunctionTransformer(max_scaling)),
        ('clustering', KMeans(n_clusters=k, random_state=42)) 
        ])
    else : 
        print("ERREUR : L'argument normalisation doit prendre la valeur 'Standard' ou la valeur 'Divmax'.")
    data_complète_2['cluster'] = pipeline.fit_predict(data_complète_2[FEATURES])
    cluster_centers_normalized = pipeline.named_steps['clustering'].cluster_centers_
    if (normalisation == "Standard"):
        cluster_centers = pipeline.named_steps['scaling'].inverse_transform(cluster_centers_normalized)
    elif (normalisation == "Divmax"):
        cluster_centers = []
        for cluster_center in cluster_centers_normalized: 
            print("Centre du clusters normalisé :", cluster_center)    
            cluster_center[0] *= maxima.iloc[0]         # On "dénormalise" manuellement les coordonnées des centres des clusters         
            cluster_center[1] *= maxima.iloc[1]         # obtenus par normalisation Divmax, car la fonction             
            cluster_center[2] *= maxima.iloc[2]         # inverse_transform ne semble pas fonctionner correctement      
            cluster_center[3] *= maxima.iloc[3]         # avec ce type de normalisation.
            cluster_center[4] *= maxima.iloc[4]
            print("Centre du clusters dénormalisé :", cluster_center)
            cluster_centers.append(cluster_center)
    cluster_centers_rounded = []
    for cluster_center in cluster_centers:      # On exprime les valeurs des features des centre des clusters
        cluster_center[0] *= 5                  # dans les unités usuelles. Aini, les durées seront exprimées
        cluster_center[1] *= 12/100             # en minutes (il y a un relevé toutes les 5 min), et les intensités
        cluster_center[2] *= 12/100             # en mm/h (et non plus en mm/5min).
        cluster_center[3] *= 144/10000
        cluster_center_rounded = [round_to_significant_figures(value, 3) for value in cluster_center]   
        # On ne garde que les 3 premiers chiffres significatifs.
        cluster_centers_rounded.append(cluster_center_rounded)
    print("Les centres des clusters sont :\n", cluster_centers_rounded)
    data_complète_3 = data_complète_2.drop('end_time_absolute', axis=1)
    print (data_complète_3)
    print(data_complète_3[['duration', 'cluster']])
    return data_complète_2, cluster_centers_rounded
    

## La fonction suivante permet de compter le nombre d'événements pluvieux de chaque cluster pour chaque pixel.
# Elle renvoie un tableau qui sera sauvegardé dans un fichier csv et permettra par la suite de représenter graphiquement les résultats du clustering réalisé ci-dessus.

def occurences_clusters(data_clusters, k):
    tableau_proportions = [[[0]*k for _ in range(300)] for _ in range(300)]
    
    # Précalculer les masques de coordonnées
    coord_i = [[i] * 300 for i in range(300)]
    coord_j = [list(range(300)) for _ in range(300)]
    
    t0 = time.time()
    for i in range(300):
        t = time.time()
        temps_restant = (t-t0)*(300-i)/(i+1)
        temps_restant_min = int(temps_restant/60)
        temps_restant_sec = int(temps_restant - temps_restant_min*60)
        print("Le programme occurences_clusters traite la ligne", i, ". Merci de patienter encore", temps_restant_min, "min", temps_restant_sec, "s.", end="\r")
        
        # Filtrer le dataframe pour les coordonnées i
        clusters_i = data_clusters[data_clusters['i'] == coord_i[i][0]]
        
        # Group by sur les colonnes 'j' et 'cluster' et compter les occurrences
        counts = clusters_i.groupby(['j', 'cluster']).size().unstack(fill_value=0)
        
        # Remplir le tableau_proportions
        for j in range(300):
            if j in counts.index:
                tableau_proportions[i][j] = counts.loc[j].tolist()
    
    print("Construction du tableau d'occurrence des clusters terminée.")
    return tableau_proportions
        



année = "2018"
k = 2

chemin_dossier_zip_1 = année + "_06.zip"
data_1 = chargement_données_zippées(chemin_dossier_zip_1)

chemin_dossier_zip_2 = année + "_07.zip"
data_2 = chargement_données_zippées(chemin_dossier_zip_2)

chemin_dossier_zip_3 = année + "_08.zip"
data_3 = chargement_données_zippées(chemin_dossier_zip_3)

chemin_dossier_zip_4 = année + "_09.zip"
data_4 = chargement_données_zippées(chemin_dossier_zip_4)
data = pd.concat([data_1, data_2, data_3, data_4])
beginning_data = data.head(10)
for index, row in beginning_data.iterrows():
    for column in beginning_data.columns:
        print(row[column])
#print(data.head(10))
méthode_normalisation = "Divmax"
rm_outliers = "yes"

data_clusters, center_clusters = clustering_KMeans(data, k, normalisation=méthode_normalisation, rm_outliers=rm_outliers)
tableau_proportions = occurences_clusters(data_clusters, k)
tableau_proportions_df = pd.DataFrame(tableau_proportions)
FEATURES = ['duration (min)', 'max_intensity (mm/h)', 'mean_intensity (mm/h)', 'variance ((mm/h)^2)', 'percentage_null']
names_features_df = pd.DataFrame(FEATURES)
header = pd.concat([names_features_df.transpose(), pd.DataFrame(center_clusters)]).reset_index(drop=True)
print(header)
df = pd.concat([header.transpose(), tableau_proportions_df]).reset_index(drop=True)
nom_fichier = "Prop_clusters\Répartition_" + str(k) + "_clusters_06-09_" + année + "_norm=" + méthode_normalisation + "_rm_outliers=" + rm_outliers + ".csv"
df.to_csv(nom_fichier, sep=',', index=True)
print ("Terminé !")


    
