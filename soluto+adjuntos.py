
from sqlalchemy import true

from functions import *

from variables import *
time1=pytime.time()
#from soluto import Soluto_up

path='./Estacionario/'
path2='./images/Soluto/'
path_s='./Soluto/'



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
t_axis=np.linspace(0,t_soluto,nt_cell)

#Contorno
Soluto_up=np.zeros((N_canals,nt_cell))
t_axis=np.linspace(0,t_soluto,nt_cell)
t_med=400
t_sig=60
Soluto_up[2]=Soluto_up[1]+1.0

Soluto_up[1]=[4.0 if i>15 else 4.0 for i in range(nt_cell)]

Soluto_up[1]=5*np.exp(-(t_axis-t_med)**2/(2*t_sig**2))


if 0:

  S_new,S_cont=soluto_forward_red(gravedad,manning,k_r,A_inicio,Q_inicio,S_inicio,Base_width,Slope_z,x_axis,nx_cell,nt_cell,Delta_x,Delta_t,Soluto_up,N_canals,matriz,plot=True,fr=freq2)

  plot_all_in_one_ws(A_inicio,Q_inicio,S_new,x_axis,Base_width,Slope_z,nt_cell,nt_cell*Delta_t,N_canals,[0,5],[0,8],'./images/Soluto/')

  gifiyer_all_in_one(path2+f'{N_canals}-canales_t',nt_cell,paso=freq2,FPS=15,gif_soluto='soluto-all.gif')



  guarda_estacionario(Base_width,Slope_z,A_inicio,Q_inicio,N_canals,path_s, mode='soluto',S=S_new)
  open(path_s+'Soluto_medidas.txt', 'w').writelines(list('\t'.join(map(str, phi)) + '\n' for phi in S_cont))

else:
  S_cont=np.zeros((N_canals,nt_cell))
  S_cont[0],S_cont[1],S_cont[2]= zip(*(map(float, line.split())
                                   for line in open(path_s+'Soluto_medidas.txt', 'r')))

time2=pytime.time()
print('t-carga:',time2-time1)

###### ADJUNTOS ######

medidas= S_cont[0] # aquí hay que seleccionar lo que se toma como medidas

###################################################Veamos el método adjunto################################################3
#medida de tiempo
plt.style.use('default')
path_3='./images/Adjuntos/'
  

#punto de partida
phi_i = [np.zeros(nx_cell[i]) for i in range(N_canals)]
phi_c = np.zeros((N_canals,nt_cell))
phi_c[2] = Soluto_up[2]
std_index=1

firewall = 100
eps_0=0.4
eps_c=eps_0


error=[]
cont = 0


#plot evolucion de la solución
fig4, ax4 = plt.subplots()
ax4.grid()
ax4.set_xlabel('t [s]')
ax4.set_ylabel('$\phi_1\,[gr\,/\,cm^3]$')
ax4.set_title('Reconstrucción Contorno red')

fig3, ax3=plt.subplots()
ax3.grid()
ax3.set_xlabel('t [s]')
ax3.set_ylabel('$\phi_m-\phi\,[gr\,/\,cm^3]$')
ax3.set_title('inyeccion')


#evolucion de la variable adjunta
fig6, ax6=plt.subplots()
ax6.set_title("$\sigma_1 (x,0)$")
ax6.set_xlabel(f'$t\;[\Delta t={Delta_t}]$')
ax6.set_ylabel(f'$\sigma$')
ax6.grid()

s_final,s_medido=soluto_forward_red(gravedad,manning,k_r,E_dif,
                                    A_inicio,Q_inicio,phi_i,
                                    Base_width,Slope_z,x_axis,
                                    nx_cell,nt_cell,Delta_x,Delta_t,
                                    phi_c,N_canals, matriz)


Diferencia = objetivo(s_medido[0],medidas)
#error.append(Diferencia)

ad_cont=np.zeros(nt_cell)


while cont<firewall and Diferencia>1e-9:# and Diferencia_old-Diferencia>-1e-6:
  
  ad_cont=evolucion_inversa_red(s_medido[0],medidas,nx_cell,nt_cell,Delta_x,Delta_t,Q_inicio,A_inicio,K=k_r,E=E_dif,N_c=N_canals,matrix=matriz)
  Diferencia_old = Diferencia


  if cont%1==0 and False:
    
    res_c = minimize_scalar(lambda x: nuevo_cont_eps_i(gravedad,manning,k_r,
                                                    A_inicio,Q_inicio,
                                                    Base_width,Slope_z,
                                                    nx_cell,nt_cell,Delta_x,Delta_t,
                                                    phi_c,phi_i,ad_ini,medidas,x)
                                                    ,bounds=(0.01,200),method='bounded')#,options={'maxiter':100,'xatol':1e-3})
  
    #print(res)
    eps_c = res_c.x
    
    

  phi_c[std_index] += eps_c*ad_cont
  #print(sum(phi_c[std_index]))

  #phi_domain = np.zeros((nx_cell,nt_cell))
 

  s_final,s_medido=soluto_forward_red(gravedad,manning,k_r,E_dif,
                                    A_inicio,Q_inicio,phi_i,
                                    Base_width,Slope_z,x_axis,
                                    nx_cell,nt_cell,Delta_x,Delta_t,
                                    phi_c,N_canals, matriz)
  
  #print('forward',fin1-start1)
  Diferencia = objetivo(s_medido[0],medidas)

  if abs(Diferencia-Diferencia_old)/Diferencia_old <0.1 and eps_c<3:
    
    if abs(Diferencia-Diferencia_old)/Diferencia_old <0.0:
      eps_c*=0.9
    else:
      eps_c+=0.2*eps_0

  if cont%15==0:
    print (Diferencia_old,Diferencia,eps_c)
    ax4.plot (t_axis,phi_c[std_index],'.',ms=2,label=f'{cont} iter')
    ax3.plot (t_axis,medidas*0-s_medido[0],'.',ms=2,label=f'{cont} iter')
    ax6.plot(t_axis,ad_cont,'.',ms=2,label=f'{cont} iter')

    #if (Diferencia_old-Diferencia)/Diferencia_old<0.2 and eps*2<0.6: eps = 2*eps

  error.append(Diferencia)
  
  cont+=1



print (Diferencia_old,Diferencia)
#error.append(Diferencia)
print('\n',cont)
ax4.plot (t_axis,phi_c[std_index],'.',color='black',ms=2.5,label=f'{cont} iter')
ax4.plot (t_axis,Soluto_up[std_index],'-',lw=1,color='r',label='t=0')
ax4.legend()

ax3.plot (t_axis,medidas*0-s_medido[0],'.',color='black',ms=2.5,label=f'{cont} iter')
ax3.legend()


#evolución del error
fig5, ax5 = plt.subplots()
ax5.grid()
ax5.set_title("error")
ax5.set_ylabel('J')
ax5.set_xlabel('iteraciones')
ax5.semilogy(error,'.-')


ax6.plot(t_axis,ad_cont,'.',color='black',ms=3,label=f'$\sigma(0,t)$ {cont} iter')
ax6.legend()

fig3.savefig(path_3+"inyeccion.jpg",dpi=200)
fig4.savefig(path_3+"reconstrucción.jpg",dpi=200)
fig5.savefig(path_3+"error.jpg",dpi=200)
fig6.savefig(path_3+"adjunto.jpg",dpi=200)
#time ejecucion

###################################SUPER-PLOT####################################################

#plot evolucion de la solución
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.style.use('ggplot')

fig7, ax7 = plt.subplots()
ax7.grid(color='black')
color='b'
ax7.set_xlabel('t [s]',color=color)
ax7.set_ylabel('$\phi\,[gr\,/\,cm^3]$',color=color)
ax7.set_title(f'Reconstrucción contorno-red de {N_canals} canales + medida')
ax7.plot (t_axis,phi_c[std_index],'.',color='black',ms=2.5)
ax7.plot (t_axis,Soluto_up[std_index],'-',lw=1,color='r')
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
#ax75.set_ylim((bottom=0,))
plt.ylim((0,5))
ax75.plot (t_axis,s_medido[0],'.',color='black',ms=2.55)
ax75.plot (t_axis,medidas,'-',lw=1,color='r')
ax75.tick_params(axis='y', labelcolor=color)
ax75.tick_params(axis='x', labelcolor=color)

fig7.savefig(path_3+"OVERVIEW_contorno.jpg",dpi=250)
