from functions import *

CFL=0.8
gravedad = 9.8
N_canals = 3

#Definir las variables del canal 

nx_cell=[]
mode=[] #puede ser 'free','up','down','both'
L=[]           
manning = []

Base_width = [[]]*N_canals
Slope_z = [[]]*N_canals

Delta_x = []
freq = 500

time = 5000

# Soluto
freq2=50
t_soluto=800
k_r=0.0



