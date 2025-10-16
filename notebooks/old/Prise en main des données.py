
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import matplotlib.animation as animation

year = 2018
month=2
day=4
path = 'C:/Users/Trist/Documents/Cours 2A - CS/Semestre 8/Projet S8/2018/'+'{:04d}/'.format(year)
file = 'RR_IDF300x300_{:04d}{:02d}{:02d}.npy'.format(year,month,day)
file = path + file
RR = np.load(file)/100.0
RR[RR < 0]=np.nan                                  # Les valeurs non définies sont codées -9999 dans les fichiers


def précipitations(year,month,day):                    # Cette fonction permet simplement de représenter l'évolutiond es précipitations en Ile-de-France sur une journée.
    path = 'C:/Users/Trist/Documents/Cours 2A - CS/Semestre 8/Projet S8/2018/'+'{:04d}/'.format(year)
    file = 'RR_IDF300x300_{:04d}{:02d}{:02d}.npy'.format(year,month,day)
    file = path + file
    RR = np.load(file)/100.0
    RR[RR < 0]=np.nan
    plt.figure()
    plt.ion()        
    for i in range (288):
        plt.imshow(RR[i,:,:], cmap='Blues')
        #plt.show(block=True)
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image', RR[i,:,:], cmap='Blues')
        plt.pause(0.2)
        plt.clf()

#dt = 1
#fig = plt.figure() # initialise la figure
#img, = plt.imshow([]) 
#plt.xlim(0, 300)
#plt.ylim(0, 300)

#def animate(i): 
#    t = i * dt
#    img.set_data(RR[i,:,:])
#    return img,
# 
#ani = animation.FuncAnimation(fig, animate, frames=100,
#                              interval=1, blit=True, repeat=False)
#plt.show()

def segmentation_événements(year,month,day):        # Cette fonction permet de séparer les différents événements de précipitations.
    path = 'C:/Users/Trist/Documents/Cours 2A - CS/Semestre 8/Projet S8/2018/'+'{:04d}/'.format(year)
    file = 'RR_IDF300x300_{:04d}{:02d}{:02d}.npy'.format(year,month,day)
    file = path + file
    RR = np.load(file)/100.0
    #RR[RR < 0]=np.nan
    RR_seg = np.zeros([288,300,300])        # RR_seg sera une carte simplifiée indiquant simplement où un événement de précpitation est en cours. C'est un tableau binaire.
    for i in range (300):
        for j in range (300):
            t = 0
            t1 = 0
            while (t<288):
                if (RR[t,i,j]>0):
                    RR_seg[t,i,j] = 1
                    t1 = 0      # Compteur du temps à partir du dernier instant de pluie. S'il atteint 6, alors l'événement de pluie est terminé.
                    t = t+1
                    t0 = t      # t0 permet de garder en mémoire le premier instant de l'acalmie.
                    while (t<288 and (RR[t,i,j]<=0) and t1<6):      # Deux averses au même endroit font partie du même événement pluvieux si elles adviennent à moins de 30 min d'intervalle.
                        t1 = t1+1
                        t = t+1
                    if (t<288 and t1<6):
                        for t2 in range (t0,t+1):
                            RR_seg[t2,i,j] = 1
                else :
                    t = t+1
    return RR_seg
                            

def affichage (carte):
    plt.figure()
    plt.ion()        
    for i in range (288):
        plt.imshow(carte[i,:,:], cmap='Reds')
        #plt.show(block=True)
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image', RR[i,:,:], cmap='Blues')
        plt.pause(0.2)
        plt.clf()       
                        


#précipitations(2018,2,1)
RR_seg = segmentation_événements(2018,2,15)
affichage(RR_seg)

