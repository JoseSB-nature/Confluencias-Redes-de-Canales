import numpy as np
from variables import *

path='./Canales/'
path2='./Estacionario/'

L,nx_cell,mode,manning,*_=inicializa(N_canals,path+'Canales.txt')

x_axis=[np.linspace(0,L[i],nx_cell[i]) for i in range(N_canals)]

if 0:

        # Canal 1
    Base_width = np.linspace(1.0,1.0,nx_cell[0])
    Slope_z = np.linspace(0.5,0.5,nx_cell[0])#+0.1*np.sin(6*np.pi*x_axis[0]/L[0])
    A = np.linspace(2.0,2.0,nx_cell[0])
    Q = np.linspace(0,0,nx_cell[0])

    open(path+'canal0.txt', 'w').writelines(list('\t'.join(map(str, med_set)) + '\n' for med_set in zip(Base_width,Slope_z,A,Q)))

        # Canal 2
    Base_width = np.linspace(1.0,1.0,nx_cell[1])
    Slope_z = np.linspace(0.5,0.5,nx_cell[1])
    A = np.linspace(2.0,2.0,nx_cell[1])
    Q = np.linspace(0,0,nx_cell[1])


    open(path+'canal1.txt', 'w').writelines(list('\t'.join(map(str, med_set)) + '\n' for med_set in zip(Base_width,Slope_z,A,Q)))

        # Canal 3
    Base_width = np.linspace(1.0,1.0,nx_cell[2])
    Slope_z = np.linspace(0.5,0.5,nx_cell[2])
    A = np.linspace(2.0,2.0,nx_cell[2])
    Q = np.linspace(0,0,nx_cell[2])


    open(path+'canal2.txt', 'w').writelines(list('\t'.join(map(str, med_set)) + '\n' for med_set in zip(Base_width,Slope_z,A,Q)))

else:
    soluto_1=np.zeros(nx_cell[0])

    soluto_2=np.array([0.0]*nx_cell[1])

    soluto_3=np.array([0.0]*nx_cell[2])

    # m=100
    # s=5
    # soluto_2=4.5*np.exp(-(m-x_axis[1])**2/(2*s**2))

    # m=25
    # s=5
    # soluto_3=1*np.exp(-(m-x_axis[2])**2/(2*s**2))

    open(path2+'Soluto_inicial.txt', 'w').writelines(list('\t'.join(map(str, phi)) + '\n' for phi in [soluto_1,soluto_2,soluto_3]))