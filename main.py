import shutil
from numpy import matrix
from functions import *
from variables import *

time_1=pytime.time()

#VACIAMOS DIRECTORIOS
path_1='./images/shallow'
path_3='./images/all-in-one'
path_2=path_1+'/animacion_'
path_4=path_3+'/animacion_'

for filename in os.listdir(path_1):
    os.remove(path_1+'/'+filename)

for filename in os.listdir(path_3):
    os.remove(path_3+'/'+filename)


path='./Canales/'


#inicialización

L, nx_cell, mode, manning, Base_width, Slope_z,A_inicio,Q_inicio,matriz,a_up,q_up,a_down,q_down = carga_red(N_canals,path)

Delta_x=[L[i]/nx_cell[i] for i in range(N_canals)]

print('Nº canales:', len(L))
print('D_x cell:', Delta_x)
print('L:', L)
print('manning:', manning)
print('mode:', mode)
#print('extremos slope(0):', Slope_z[0][0],Slope_z[1][0],Slope_z[2][0])
print('red:matrix:',matriz)

x_axis=[np.linspace(0,L[i],nx_cell[i]) for i in range(N_canals)]
t=1
real_t=0
Range_A=[0,4]
Range_Q=[0,8]
mac=0

A_old=A_inicio
Q_old=Q_inicio


# plot_perfil_soluto_red(A_old,Q_old,
#                     x_axis,Base_width,Slope_z,
#                     t,real_t,N_canals,
#                     Range_A,Range_Q,path_2)

plot_all_in_one(A_old,Q_old,
                    x_axis,Base_width,Slope_z,
                    t,real_t,N_canals,
                    Range_A,Range_Q,path_4)


while real_t<time:
    time_1=pytime.time()
    A_new,Q_new,D_t = update_red(gravedad,manning,
                                A_old,Q_old,
                                Base_width,Slope_z,
                                nx_cell,N_canals,CFL,Delta_x,
                                mode,
                                a_up,q_up,a_down,q_down,
                                matriz,time,real_t)
    #Guardamos la malla
    real_t+=D_t
    if t%freq==0:
        # plot_perfil_soluto_red(A_new,Q_new,
        #                 x_axis,Base_width,Slope_z,
        #                 t+1,real_t,N_canals,
        #                 Range_A,Range_Q,path_2)
        print(real_t)
        plot_all_in_one(A_old,Q_old,
                    x_axis,Base_width,Slope_z,
                    t,real_t,N_canals,
                    Range_A,Range_Q,path_4)
    
        plt.close('all')

    #Esta=estabilidad(Q_old,Q_new)<tolerancia

    


    A_old=A_new.copy()
    Q_old=Q_new.copy()

    time_2=pytime.time()
    if t==1:print(f"tiempo evolución:{time_2-time_1:.4f}s")

    

    t+=1



# plot_perfil_soluto_red(A_old,Q_old,
#                     x_axis,Base_width,Slope_z,
#                     t,real_t,N_canals,
#                     Range_A,Range_Q,path_2)

plot_all_in_one(A_old,Q_old,
                    x_axis,Base_width,Slope_z,
                    t,real_t,N_canals,
                    Range_A,Range_Q,path_4)

#gifiyer_red(path_2,t,N_canals,paso=freq,FPS=10)

gifiyer_all_in_one(path_4+f'{N_canals}-canales_t',t,paso=freq,FPS=10)


#Guardamos estacionario
path_state='./Estacionario/'
shutil.rmtree(path_state[:-1])
shutil.copytree(path[:-1],path_state[:-1])
guarda_estacionario(Base_width,Slope_z,A_new,Q_new,N_canals,path_state)