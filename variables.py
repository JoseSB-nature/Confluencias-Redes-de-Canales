from functions import *

CFL=1.0
peclet=0.4
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
freq = 25

time = 100

def contornos(t_r,t_t):
    a_up = [1,
            1,
            1] 

    q_up = [0,
            1,
            1+0.0*np.sin(t_r*2*np.pi*20/t_t)] 

    a_down=[2.0,
            2.0,
            2.0] 

    q_down=[1,
            1,
            1] 

    return a_up,q_up,a_down,q_down


# Soluto
k_r=0.0
E_dif=0

