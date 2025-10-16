
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import io
import time
import os
import ast


# Petite fonction préliminaire qui convertit une chaîne de caractère représentant une liste en liste.
def str_to_list(s):
    return ast.literal_eval(s)

def affichage_répartition_clusters(tableau_proportions, month, year, color = 'gray'):
    tableau_prop = tableau_proportions.iloc[:,1:]   # On enlève la première colonne, composée des indices de localisation.
    tableau_prop_lst = tableau_prop.applymap(str_to_list)   # On transforme les chaînes de caractères en listes.
    list_of_lists = tableau_prop_lst.values.tolist()    # On convertit le dataframe en liste de listes.
    # On vérifie si toutes les listes ont la même longueur.
    if len(set(map(len, list_of_lists))) != 1:
        print("Les listes n'ont pas toutes la même longueur.")
    else:
        tableau_prop_lst = np.array(list_of_lists)  # Si tout va bien, on convertit la liste de listes en tableau numpy.
    print(tableau_prop_lst)
    shape = tableau_prop_lst.shape
    print(shape)
    k = tableau_prop_lst.shape[2]   # k représente le nombre de clusters.
    plt.figure(figsize = (10,8))
    if (k == 2):
        sums = tableau_prop_lst.sum(axis=2)
        carte_répartition = tableau_prop_lst[:,:,1] / sums      
        print(carte_répartition)
        plt.imshow(carte_répartition, cmap=color)
    elif (k == 3):
        somme = np.sum(tableau_prop_lst, axis=2, keepdims=True)
        carte_répartition = (256 * tableau_prop_lst / somme).astype(np.uint8)
        print(carte_répartition)
        plt.imshow(carte_répartition)
        plt.colorbar()
    titre = "Carte géographique de la répartition des différents clusters pour un nombre de clusters égal à " + str(k)
    plt.title(titre)
    sous_titre = month + "/" + year
    plt.suptitle(sous_titre)
    plt.colorbar(label="Proportion d'événements dans le cluster 1")
    plt.show()


def affichage_répartition_clusters_2(tableau_proportions, month, year, color = 'gray'):
    tableau_prop = tableau_proportions.iloc[:,1:]   # On enlève la première colonne, composée des indices de localisation.
    tableau_prop_lst = tableau_prop.applymap(str_to_list)   # On transforme les chaînes de caractères en listes.
    list_of_lists = tableau_prop_lst.values.tolist()    # On convertit le dataframe en liste de listes.
    # On vérifie si toutes les listes ont la même longueur.
    if len(set(map(len, list_of_lists))) != 1:
        print("Les listes n'ont pas toutes la même longueur.")
    else:
        tableau_prop_lst = np.array(list_of_lists)  # Si tout va bien, on convertit la liste de listes en tableau numpy.
    print(tableau_prop_lst)
    shape = tableau_prop_lst.shape
    print(shape)
    k = tableau_prop_lst.shape[2]   # k représente le nombre de clusters.
    fig, axes = plt.subplots(1+(k-1)//3, 3, figsize=(15, 7))
    sums = tableau_prop_lst.sum(axis=2)
    if (k<=3):
        for l in range (k):
            carte_répartition = tableau_prop_lst[:,:,l] / sums
            im = axes[l].imshow(carte_répartition, cmap = color)
            axes[l].set_title('Répartition du cluster {}'.format(l))
            cbar = fig.colorbar(im, ax=axes[l], shrink=0.8)
            cbar.set_label("Proportion d'événements dans le cluster {}".format(l), size='medium')
            size_colorbar = cbar.ax.get_position().height
            cbar.ax.yaxis.label.set_size(size_colorbar * 0.5)
    else : 
        for l in range (k):
            carte_répartition = tableau_prop_lst[:,:,l] / sums
            im = axes[l//3, l%3].imshow(carte_répartition, cmap = color)
            axes[l//3, l%3].set_title('Répartition du cluster {}'.format(l))
            cbar = fig.colorbar(im, ax=axes[l//3, l%3])
            cbar.set_label("Proportion d'événements dans le cluster {}".format(l)) 
    titre = "Cartes géographiques de la répartition des différents clusters pour un nombre de clusters égal à " + str(k)
    fig.suptitle(titre)
    sous_titre = month + "/" + year
    fig.text(0.5, 0.95, sous_titre, ha='center', va='top')
    plt.tight_layout()
    plt.show()

    
mois = "06-09"
année = "2018"
nombre_clusters = 5
méthode_normalisation = "Divmax"
rm_outliers = "yes"
#chemin_proportions = ".\Répartition_2_clusters_" + mois + "_" + année + ".csv"
#chemin_proportions = ".\Répartition_" + str(nombre_clusters) + "_clusters_06-09_" + année + "_norm=" + méthode_normalisation + ".csv"
#chemin_proportions = ".\Répartition_" + str(nombre_clusters) + "_clusters_06-09_" + année + ".csv"
chemin_proportions = ".\Répartition_" + str(nombre_clusters) + "_clusters_06-09_" + année + "_norm=" + méthode_normalisation + "rm_outliers=" + rm_outliers + ".csv"
tableau_proportions_df = pd.read_csv(chemin_proportions)
#print(tableau_proportions_df)
affichage_répartition_clusters_2(tableau_proportions_df, mois, année)
    
    
