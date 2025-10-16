import numpy as np
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import random as rd
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
import matplotlib.colors as cls
import csv
import ast
from scipy.signal import convolve2d, convolve




#Utilisation des données téléchargées
#Récupération des indices des données manquantes
n = 10
def create_data_array(directory) :
    datalist = []
    for i in range(n):
        filename = os.listdir(directory)[i]
        if filename.endswith(".npy"):
            data_day = np.load(directory + '/' + filename)
        datalist.append(data_day)
    data_raw = np.concatenate(datalist,axis=0)
    return data_raw


data_raw = create_data_array(r'datatest')

time_grad = data_raw[1:,:,:]-data_raw[:-1,:,:]

def csv_to_list(csv_file_path):
    data_list = []
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        
        for row in csv_reader:
            row_list = [[int(num) for num in string.strip('[]').split()] for string in row]
            print(row_list)
            data_list.append(row_list)
    return data_list


def grad_horizontal(A):
    return A[:,1:,:]-A[:,:-1,:]

def grad_vertical(A):
    return A[:,:,1:]-A[:,:,:-1]

gradh=grad_horizontal(data_raw)
gradv=grad_vertical(data_raw)

def animation(data):
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0])

    cbar = fig.colorbar(im)
    cbar.set_label('Value')


    def init():
        im.set_data(data[0])
        return [im]

    def update(frame):
        im.set_data(data[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

    plt.show()

def animation_accumulated(data):
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0],cmap='gray')
    cbar = fig.colorbar(im)
    cbar.set_label('Value')
    def init():
        im.set_data(data[0])
        return [im]

    def update(frame):
        im.set_data(np.sum(data[:frame,:,:],axis=0))
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

    plt.show()

def animation_download(data,filename):
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0])
    cbar = fig.colorbar(im)
    cbar.set_label('Value')

    def init():
        im.set_data(data[0])
        return [im]

    def update(frame):
        im.set_data(data[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    ani.save(f'animation_{filename}.gif', writer='pillow', fps=1,dpi=80)
    plt.colorbar(im, ax=ax, label='rain level')

def animation_accumulated_download(data,filename):
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0])
    cbar = fig.colorbar(im)
    cbar.set_label('Value')

    def init():
        im.set_data(data[0])
        return [im]

    def update(frame):
        im.set_data(np.sum(data[:frame,:,:],axis=0))
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    ani.save(f'animation_{filename}.gif', writer='pillow', fps=30,dpi=80)
    plt.colorbar(im, ax=ax, label='rain accumulation')


def animation_circle(data,outliers_list):
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0])

    cbar = fig.colorbar(im)
    cbar.set_label('Value')


    def init():
        im.set_data(data[0])
        for element in outliers_list[0]:
            circle = plt.Circle(element, radius=5, edgecolor='white', facecolor='none')
            ax.add_patch(circle)

        return [im]

    def update(frame):
        im.set_data(data[frame])
        if outliers_list[frame]!=None:
            for element in outliers_list[frame]:
                x=element[1]
                y=element[0]
                circle = plt.Circle((x,y), radius=5, edgecolor='white', facecolor='none')
                ax.add_patch(circle)
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

    plt.show()

def animation_circle_download(data,outliers_list,filename):
    num_frames = data.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(data[0])

    cbar = fig.colorbar(im)
    cbar.set_label('Value')


    def init():
        im.set_data(data[0])
        for element in outliers_list[0]:
            circle = plt.Circle(element, radius=5, edgecolor='white', facecolor='none')
            ax.add_patch(circle)

        return [im]

    def update(frame):
        im.set_data(data[frame])
        if outliers_list[frame]!=None:
            for element in outliers_list[frame]:
                x=element[1]
                y=element[0]
                circle = plt.Circle((x,y), radius=5, edgecolor='white', facecolor='none')
                ax.add_patch(circle)
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    ani.save(f'animation_{filename}.gif', writer='pillow', fps=30,)
    

#outlierlist = csv_to_list('raw_data_output.csv')

# animation_circle(data_raw,outlierlist)
kernel_2D = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
data_4_hours = np.array([np.sum(data_raw[48*i:48*i+48],axis=0) for i in range(len(data_raw)//48)])
S = np.array([convolve2d(element,kernel_2D,mode='same') for element in data_4_hours])
animation_download(S,'S_four_hours')
animation_download(data_4_hours,'4_hours')