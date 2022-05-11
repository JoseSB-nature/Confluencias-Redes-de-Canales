from functions import *
from variables import *

path='./Estacionario/'
path2='./images/Soluto/'

for filename in os.listdir(path2):
    os.remove(path2[:-1]+'/'+filename)

L, nx_cell, mode, manning, Base_width, Slope_z,A_inicio,Q_inicio,matriz,a_up,q_up,a_down,q_down = carga_red(N_canals,path)

Delta_x=[L[i]/nx_cell[i] for i in range(N_canals)]

Delta_t=min([calcula_dt(gravedad, np.array(A_inicio[j]), np.array(Q_inicio[j]), Base_width[j], nx_cell[j], CFL, Delta_x[j]) for j in range(N_canals)])

nt_cell=int(t_soluto/Delta_t)

print (Delta_x,Delta_t)
print (nx_cell,nt_cell)

S_inicio=[np.zeros(nx_cell[i]) for i in range(N_canals)]
with open(path+'Soluto_inicial.txt') as g:
        for i,line in enumerate(g):
            linea=[float(i) for i in line.split()]
            S_inicio[i]=linea
S_inicio=np.array(S_inicio)

x_axis=[np.linspace(0,L[i],nx_cell[i]) for i in range(N_canals) ]

#Contorno
Soluto_up=np.zeros((N_canals,nt_cell))
t_axis=np.linspace(0,t_soluto,nt_cell)
t_med=60
t_sig=10

Soluto_up[1]=Soluto_up[1]+0.0

Soluto_up[2]=[4.0 if i>15 else 0.0 for i in range(nt_cell)]

Soluto_up[2]=5*np.exp(-(t_axis-t_med)**2/(2*t_sig**2))

S_new,S_cont=soluto_forward_red(gravedad,manning,k_r,A_inicio,Q_inicio,S_inicio,Base_width,Slope_z,x_axis,nx_cell,nt_cell,Delta_x,Delta_t,Soluto_up,N_canals,matriz,plot=True,fr=freq2)

plot_all_in_one_ws(A_inicio,Q_inicio,S_new,x_axis,Base_width,Slope_z,nt_cell,nt_cell*Delta_t,N_canals,[0,5],[0,8],'./images/Soluto/')

gifiyer_all_in_one(path2+f'{N_canals}-canales_t',nt_cell,paso=freq2,FPS=15,gif_soluto='soluto-all.gif')

path_s='./Soluto/'

guarda_estacionario(Base_width,Slope_z,A_inicio,Q_inicio,N_canals,path_s, mode='soluto',S=S_new)
open(path_s+'Soluto_medidas.txt', 'w').writelines(list('\t'.join(map(str, med_set)) + '\n' for med_set in zip(*[med for med in S_cont])))
