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
t_med=20
t_sig=5

Soluto_up[1]=Soluto_up[1]+4.0

Soluto_up[2]=[4.0 if i>15 else 0.0 for i in range(nt_cell)]

#Soluto_up[2]=2*np.exp(-(t_axis-t_med)**2/(2*t_sig**2))

S_new,S_cont=soluto_forward_red(gravedad,manning,k_r,A_inicio,Q_inicio,S_inicio,Base_width,Slope_z,x_axis,nx_cell,nt_cell,Delta_x,Delta_t,Soluto_up,N_canals,matriz,plot=True,fr=freq2)

plot_all_in_one_ws(A_inicio,Q_inicio,S_new,x_axis,Base_width,Slope_z,nt_cell,nt_cell*Delta_t,N_canals,[0,5],[0,8],'./images/Soluto/')

gifiyer_all_in_one(path2+f'{N_canals}-canales_t',nt_cell,paso=freq2,FPS=15,gif_soluto='soluto-all.gif')

path_s='./Soluto/'

guarda_estacionario(Base_width,Slope_z,A_inicio,Q_inicio,N_canals,path_s, mode='soluto',S=S_new)
open(path_s+'Soluto_medidas.txt', 'w').writelines(list('\t'.join(map(str, phi)) + '\n' for phi in S_cont))
###### ADJUNTOS ######

medidas= S_cont[0] # aquí hay que seleccionar lo que se toma como medidas

###################################################Veamos el método adjunto################################################3
#medida de tiempo
plt.style.use('default')

start1 = pytime.time()
  

#punto de partida
phi_i = np.zeros(nx_cell)
phi_c = np.zeros(nt_cell)

firewall = 100
eps_i=0.4
eps_c=0.4

Diferencia = 1e3
Diferencia_old = Diferencia+1
error=[]
cont = 0

medidas=Soluto_contorno.copy()

time_up = t_river

#plot evolucion de la solución
fig4, ax4 = plt.subplots()
ax4.grid()
ax4.set_xlabel('x [m]')
ax4.set_ylabel('$\phi\,[gr\,/\,cm^3]$')
ax4.set_title('transporte convectivo de flujo variable')


#evolucion de la variable adjunta
fig6, ax6=plt.subplots()
ax6.set_title("$\sigma (x,0)$")
ax6.set_xlabel(f'$x\;[\Delta x={Delta_x}]$')
ax6.set_ylabel(f'$\sigma$')
ax6.grid()
t_1=pytime.time()
s_final,s_contorno=soluto_forward(gravedad,manning,k_r,
                                    A_river,Q_river,phi_i,
                                    Base_river,Slope_river,
                                    nx_cell,nt_cell,Delta_x,Delta_t,
                                    s_up_river)
t_2=pytime.time()
print('adjunto forward:',t_2-t_1)
Diferencia = objetivo(s_contorno,medidas)
#error.append(Diferencia)

ad_cont=np.zeros(nt_cell)
ad_ini=np.zeros(nx_cell)

while cont<firewall and Diferencia>1e-9:# and Diferencia_old-Diferencia>-1e-6:
  t_1=pytime.time()
  ad_ini,ad_cont=evolucion_inversa(s_contorno,medidas,nx_cell,nt_cell,Delta_x,Delta_t,Q_river,A_river,K=k_r)
  Diferencia_old = Diferencia
  t_2=pytime.time()
  if cont<3:print('adjunto backward:',t_2-t_1)

  if cont%1==0:
    t_1=pytime.time()
    res_i = minimize_scalar(lambda x: nuevo_cont_eps_i(gravedad,manning,k_r,
                                                    A_river,Q_river,
                                                    Base_width,Slope_z,
                                                    nx_cell,nt_cell,Delta_x,Delta_t,
                                                    phi_c,phi_i,ad_ini,medidas,x)
                                                    ,bounds=(0.01,200),method='bounded')#,options={'maxiter':100,'xatol':1e-3})
    #print(res)
    res_c = minimize_scalar(lambda x: nuevo_cont_eps_c(gravedad,manning,k_r,
                                                    A_river,Q_river,
                                                    Base_width,Slope_z,
                                                    nx_cell,nt_cell,Delta_x,Delta_t,
                                                    phi_c,phi_i,ad_cont,medidas,x)
                                                    ,bounds=(0.01,200),method='bounded')#,options={'maxiter':100,'xatol':1e-3})
    #print(res)
    eps_i = res_i.x
    eps_c = res_c.x
    t_2=pytime.time()
    if cont<3:print('adjunto golden:',t_2-t_1) 
  else:
    eps_i=eps_c=0.4

  phi_i += eps_i*ad_ini
  phi_c += eps_c*ad_cont

  #phi_domain = np.zeros((nx_cell,nt_cell))
 
  t_1=pytime.time()
  s_final,s_contorno=soluto_forward(gravedad,manning,k_r,A_river,Q_river,phi_i,Base_width,Slope_z,nx_cell,nt_cell,Delta_x,Delta_t,phi_c)
  t_2=pytime.time()
  if cont<3:print('adjunto forward:',t_2-t_1)
  #print('forward',fin1-start1)
  Diferencia = objetivo(s_contorno,medidas)

  
  if cont%25==0:
    print (Diferencia_old,Diferencia,eps_i)
    ax4.plot (x_river,phi_i,'.',ms=2.5,label=f'{cont} iter')
    ax6.plot(x_river,ad_ini,'.',ms=2.5,label=f'{cont} iter')

    #if (Diferencia_old-Diferencia)/Diferencia_old<0.2 and eps*2<0.6: eps = 2*eps

  error.append(Diferencia)
  
  cont+=1

fin1= pytime.time()
print('adjuntos:',fin1-start1)

print (Diferencia_old,Diferencia)
#error.append(Diferencia)
print('\n',cont)
ax4.plot (x_river,phi_i,'.',color='black',ms=2.5,label=f'{cont} iter')
ax4.plot (x_river,S_inicio,'-',lw=1,color='r',label='t=0')
ax4.legend()



#evolución del error
fig5, ax5 = plt.subplots()
ax5.grid()
ax5.set_title("error")
ax5.set_ylabel('J')
ax5.set_xlabel('iteraciones')
ax5.semilogy(error,'.-')


ax6.plot(x_river,ad_ini,'.',color='black',ms=3,label=f'$\sigma(0,t)$ {cont} iter')
ax6.legend()

fig4.savefig(path_3+"/reconstrucción.jpg",dpi=200)
fig5.savefig(path_3+"/error.jpg",dpi=200)
fig6.savefig(path_3+"/adjunto.jpg",dpi=200)
#time ejecucion

###################################SUPER-PLOT####################################################

#plot evolucion de la solución
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use('ggplot')

fig7, ax7 = plt.subplots()
ax7.grid(color='black')
color='b'
ax7.set_xlabel('x [m]',color=color)
ax7.set_ylabel('$\phi\,[gr\,/\,cm^3]$',color=color)
ax7.set_title('transporte convectivo de flujo variable:inicial')
ax7.plot (x_river,phi_i,'.',color='black',ms=2.5)
ax7.plot (x_river,S_inicio,'-',lw=1,color='r')
ax7.tick_params(axis='y', labelcolor=color)
ax7.tick_params(axis='x', labelcolor=color)
plt.style.use('seaborn-bright')
ax75 = inset_axes(ax7,
                width="45%", # width = 30% of parent_bbox
                height=1.5, # height : 1 inch
                loc='right')

ax75.grid(color='black')
color='black'
ax75.set_xlabel('t [s]',color=color)
ax75.set_ylabel('$\phi\,[gr\,/\,cm^3]$',color=color)
ax75.set_title('medidas contorno')
ax75.plot (t_river,s_contorno,'.',color='black',ms=2.55)
ax75.plot (t_river,medidas,'-',lw=1,color='r')
ax75.tick_params(axis='y', labelcolor=color)
ax75.tick_params(axis='x', labelcolor=color)

fig7.savefig(path_3+"/OVERVIEW_inicio.jpg",dpi=250)

#plot evolucion de la solución
plt.clf()
plt.style.use('ggplot')

fig8, ax8 = plt.subplots()
ax8.grid(color='black')
color='b'
ax8.set_xlabel('x [m]',color=color)
ax8.set_ylabel('$\phi\,[gr\,/\,cm^3]$',color=color)
ax8.set_title('transporte convectivo de flujo variable:contorno')
ax8.plot (t_river,phi_c,'.',color='black',ms=2.5)
ax8.plot (t_river,s_up_river,'-',lw=1,color='r')
ax8.tick_params(axis='y', labelcolor=color)
ax8.tick_params(axis='x', labelcolor=color)
plt.style.use('seaborn-bright')
ax85 = inset_axes(ax8,
                width="45%", # width = 30% of parent_bbox
                height=1.5, # height : 1 inch
                loc='right')

ax85.grid(color='black')
color='black'
ax85.set_xlabel('t [s]',color=color)
ax85.set_ylabel('$\phi\,[gr\,/\,cm^3]$',color=color)
ax85.set_title('medidas contorno')
ax85.plot (t_river,s_contorno,'.',color='black',ms=2.55)
ax85.plot (t_river,medidas,'-',lw=1,color='r')
ax85.tick_params(axis='y', labelcolor=color)
ax85.tick_params(axis='x', labelcolor=color)

fig8.savefig(path_3+"/OVERVIEW_contorno.jpg",dpi=250)