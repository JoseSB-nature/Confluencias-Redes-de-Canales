from http.client import NO_CONTENT
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time as pytime
from numba import njit, jit
from IPython import display
import os
from scipy.optimize import minimize_scalar
import imageio

#plt.style.use('dark_background')


############################### Shallow-water #############################################
@jit
def R_hidraulico(A, Bw, i):
    return A[i]/(Bw[i]+2*A[i]/Bw[i])


@jit
def S_manning(Q, A, Bw, n, i):
    Rh = R_hidraulico(A, Bw, i)
    return (Q[i]*n)**2/(A[i]**2*Rh**(4.0/3.0))


@jit
def c_medio(g, A, Bw, i):
    return np.sqrt(g*(A[i]+A[i+1])/(Bw[i]+Bw[i+1]))


@jit
def u_medio(A, Q, i):
    u = Q/A
    return(u[i]*np.sqrt(A[i])+u[i+1]*np.sqrt(A[i+1]))/(np.sqrt(A[i])+np.sqrt(A[i+1]))


@jit
def lambda1(g, A, Q, Bw, i):
    return u_medio(A, Q, i) - c_medio(g, A, Bw, i)


@jit
def lambda2(g, A, Q, Bw, i):
    return u_medio(A, Q, i) + c_medio(g, A, Bw, i)


@jit
def avec_1(g, A, Q, Bw, i):
    return np.array([1.0, lambda1(g, A, Q, Bw, i)])


@jit
def avec_2(g, A, Q, Bw, i):
    return np.array([1.0, lambda2(g, A, Q, Bw, i)])


@jit
def alpha1(g, A, Q, Bw, i):
    return(lambda2(g, A, Q, Bw, i)*(A[i+1]-A[i])-(Q[i+1]-Q[i]))/(2*c_medio(g, A, Bw, i))


@jit
def alpha1_fix(g, A, Q, Bw, i, lambda_2):
    return(lambda_2*(A[i+1]-A[i])-(Q[i+1]-Q[i]))/(2*c_medio(g, A, Bw, i))


@jit
def alpha2(g, A, Q, Bw, i):
    return(-lambda1(g, A, Q, Bw, i)*(A[i+1]-A[i])+(Q[i+1]-Q[i]))/(2*c_medio(g, A, Bw, i))


@jit
def alpha2_fix(g, A, Q, Bw, i, lambda_1):
    return(-lambda_1*(A[i+1]-A[i])+(Q[i+1]-Q[i]))/(2*c_medio(g, A, Bw, i))


@jit
def beta1(g, n, Delta_x, A, Q, Bw, slope, i):
    c_ = c_medio(g, A, Bw, i)
    A_ = 0.5*(A[i]+A[i+1])
    B_ = 0.5*(Bw[i]+Bw[i+1])
    S0_ = -(slope[i+1]-slope[i])/Delta_x
    # Cambiar si se añade manning
    Sf_ = 0.5*(S_manning(Q, A, Bw, n, i)+S_manning(Q, A, Bw, n, i+1))
    dA = A[i+1]-A[i]
    dQ = Q[i+1]-Q[i]
    dh = A[i+1]/Bw[i+1]-A[i]/Bw[i]
    return -(g*A_*((S0_-Sf_)*Delta_x-dh+dA/B_))/(2*c_)


@jit
def beta2(g, n, Delta_x, A, Q, Bw, slope, i):
    return -beta1(g, n, Delta_x, A, Q, Bw, slope, i)


@jit
def gamma1(g, n, Delta_x, A, Q, Bw, slope, i):
    return alpha1(g, A, Q, Bw, i)-beta1(g, n, Delta_x, A, Q, Bw, slope, i)/lambda1(g, A, Q, Bw, i)

@jit
def gamma2(g, n, Delta_x, A, Q, Bw, slope, i):
    return alpha2(g, A, Q, Bw, i)-beta2(g, n, Delta_x, A, Q, Bw, slope, i)/lambda2(g, A, Q, Bw, i)

@jit
def lambda_i(g, A, Q, Bw, i, mode):  # mode selecciona el autovalor
    u = Q[i]/A[i]
    c = np.sqrt(g*A[i]/Bw[i])
    return u+(-1)**mode*c


@jit
def check_fix(g, A, Q, Bw, i, mode):  # comprueba i,i+1

    return (lambda_i(g, A, Q, Bw, i, mode)*lambda_i(g, A, Q, Bw, i+1, mode)) < 0


@jit
def l_bar(g, A, Q, Bw, i, mode):
    l_i = lambda_i(g, A, Q, Bw, i, mode)
    l_j = lambda_i(g, A, Q, Bw, i+1, mode)
    l_med = lambda1(g, A, Q, Bw, i) if mode == 1 else lambda2(g, A, Q, Bw, i)
    return l_i*(l_j-l_med)/(l_j-l_i)


@jit
def l_hat(g, A, Q, Bw, i, mode):
    l_i = lambda_i(g, A, Q, Bw, i, mode)
    l_j = lambda_i(g, A, Q, Bw, i+1, mode)
    l_med = lambda1(g, A, Q, Bw, i) if mode == 1 else lambda2(g, A, Q, Bw, i)
    return l_j*(l_med-l_i)/(l_j-l_i)


@jit
def calcula_dt(g, A, Q, Bw, nx_cell, CFL, Delta_x):
    lambdas = []
    for i in range(nx_cell-1):
        if A[i] != 0 or A[i+1] != 0:
            lambdas.append(abs(lambda1(g, A, Q, Bw, i)))
            lambdas.append(abs(lambda2(g, A, Q, Bw, i)))

    lambdas_clear=[x for x in lambdas if np.isnan(x)==False]
    #for i,a in enumerate(lambdas): 
        #if np.isnan(a)==True: 
            #print('nan',i)
    res = CFL*Delta_x/max(lambdas_clear)

    return res


@jit
def flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, i):
    ans = 0.0
    lambda1_m = 0.5*(lambda1(g, A, Q, Bw, i)-abs(lambda1(g, A, Q, Bw, i)))
    lambda2_m = 0.5*(lambda2(g, A, Q, Bw, i)-abs(lambda2(g, A, Q, Bw, i)))

    if lambda1_m != 0:
        ans = ans + lambda1_m*gamma1(g, n, Delta_x, A, Q, Bw, slope, i)

    if lambda2_m != 0:
        ans = ans + lambda2_m*gamma2(g, n, Delta_x, A, Q, Bw, slope, i)
    return Q[i] + ans


@jit
def lista_flujos(g: float, n: float,
                 A: np.ndarray, Q: np.ndarray, S: np.ndarray,
                 Bw: np.ndarray, slope: np.ndarray,
                 nx_cell, Delta_x):

    flujos_1 = np.zeros((nx_cell, 3))
    flujos_2 = np.zeros((nx_cell, 3))
    flujos_num = np.zeros(nx_cell)  # Q.copy()
    for i in range(nx_cell):
        uno = dos = False
        if i < (nx_cell-1):
            # OJO####### 1 y 2 pasan a ser left y right
            uno = check_fix(g, A, Q, Bw, i, 1)
            dos = check_fix(g, A, Q, Bw, i, 2)

        if uno or dos:
            #print(f'boom: ({i})')

            if i > 0 and i < (nx_cell-1):
                lambda1_hat = l_hat(g, A, Q, Bw, i, 1)
                lambda2_hat = l_hat(g, A, Q, Bw, i, 2)
                lambda1_bar = l_bar(g, A, Q, Bw, i, 1)
                lambda2_bar = l_bar(g, A, Q, Bw, i, 2)
                lambda1_i = lambda1(g, A, Q, Bw, i)
                lambda2_i = lambda2(g, A, Q, Bw, i)
                gamma1_hat = alpha1_fix(g, A, Q, Bw, i, lambda2_hat)
                gamma2_hat = alpha2_fix(g, A, Q, Bw, i, lambda1_hat)
                gamma1_bar = alpha1_fix(
                    g, A, Q, Bw, i, lambda2_bar)-beta1(g, n, Delta_x, A, Q, Bw, slope, i)/lambda1_bar
                gamma2_bar = alpha2_fix(
                    g, A, Q, Bw, i, lambda1_bar)-beta2(g, n, Delta_x, A, Q, Bw, slope, i)/lambda2_bar

            if uno and not dos:

                flujos_1[i, 0] = (np.sign(lambda1_hat) > 0)*lambda1_hat*gamma1_hat +\
                    (np.sign(lambda1_bar) > 0)*lambda1_bar*gamma1_bar +\
                    (np.sign(lambda2_i) > 0)*lambda2_i * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i)
                flujos_2[i, 0] = (np.sign(lambda1_hat) < 0)*lambda1_hat*gamma1_hat +\
                    (np.sign(lambda1_bar) < 0)*lambda1_bar*gamma1_bar +\
                    (np.sign(lambda2_i) < 0)*lambda2_i * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i)
                flujos_1[i, 1] = (np.sign(lambda1_hat) > 0)*lambda1_hat*2*gamma1_hat +\
                    (np.sign(lambda1_bar) > 0)*lambda1_bar**2*gamma1_bar +\
                    (np.sign(lambda2_i) > 0)*lambda2_i**2 * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i)
                flujos_2[i, 1] = (np.sign(lambda1_hat) < 0)*lambda1_hat*2*gamma1_hat +\
                    (np.sign(lambda1_bar) < 0)*lambda1_bar**2*gamma1_bar +\
                    (np.sign(lambda2_i) < 0)*lambda2_i**2 * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i)

            if dos and not uno:

                flujos_1[i, 0] = (np.sign(lambda2_hat) > 0)*lambda2_hat*gamma2_hat +\
                    (np.sign(lambda2_bar) > 0)*lambda2_bar*gamma2_bar +\
                    (np.sign(lambda1_i) > 0)*lambda1_i * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i)
                flujos_2[i, 0] = (np.sign(lambda2_hat) < 0)*lambda2_hat*gamma2_hat +\
                    (np.sign(lambda2_bar) < 0)*lambda2_bar*gamma2_bar +\
                    (np.sign(lambda1_i) < 0)*lambda1_i * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i)

                flujos_1[i, 1] = (np.sign(lambda2_hat) > 0)*lambda2_hat**2*gamma2_hat +\
                    (np.sign(lambda2_bar) > 0)*lambda2_bar**2*gamma2_bar +\
                    (np.sign(lambda1_i) > 0)*lambda1_i**2 * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i)
                flujos_2[i, 1] = (np.sign(lambda2_hat) < 0)*lambda2_hat**2*gamma2_hat +\
                    (np.sign(lambda2_bar) < 0)*lambda2_bar**2*gamma2_bar +\
                    (np.sign(lambda1_i) < 0)*lambda1_i**2 * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i)

            if uno and dos:

                flujos_1[i, 0] = (np.sign(lambda1_hat) > 0)*lambda1_hat*gamma1_hat +\
                    (np.sign(lambda1_bar) > 0)*lambda1_bar*gamma1_bar +\
                    (np.sign(lambda2_hat) > 0)*lambda2_hat*gamma2_hat +\
                    (np.sign(lambda2_bar) > 0)*lambda2_bar*gamma2_bar
                flujos_2[i, 0] = (np.sign(lambda1_hat) < 0)*lambda1_hat*gamma1_hat +\
                    (np.sign(lambda1_bar) < 0)*lambda1_bar*gamma1_bar +\
                    (np.sign(lambda2_hat) < 0)*lambda2_hat*gamma2_hat +\
                    (np.sign(lambda2_bar) < 0)*lambda2_bar*gamma2_bar

                flujos_1[i, 1] = (np.sign(lambda1_hat) > 0)*lambda1_hat*gamma1_hat +\
                    (np.sign(lambda1_bar) > 0)*lambda1_bar**2*gamma1_bar +\
                    (np.sign(lambda2_hat) > 0)*lambda2_hat**2*gamma2_hat +\
                    (np.sign(lambda2_bar) > 0)*lambda2_bar**2*gamma2_bar
                flujos_2[i, 1] = (np.sign(lambda1_hat) < 0)*lambda1_hat*gamma1_hat +\
                    (np.sign(lambda1_bar) < 0)*lambda1_bar**2*gamma1_bar +\
                    (np.sign(lambda2_hat) < 0)*lambda2_hat**2*gamma2_hat +\
                    (np.sign(lambda2_bar) < 0)*lambda2_bar**2*gamma2_bar

        else:

            if i == nx_cell-1:

                A_prima = np.append(A, A[-1])
                Q_prima = np.append(Q, Q[-1])
                Bw_prima = np.append(Bw, Bw[-1])
                slope_prima = np.append(slope, slope[-1])

                lambda1_i = lambda1(g, A_prima, Q_prima, Bw_prima, i)
                lambda2_i = lambda2(g, A_prima, Q_prima, Bw_prima, i)
                flujos_1[i, 0] = lambda1_i * gamma1(g, n, Delta_x, A_prima, Q_prima,
                                                    Bw_prima, slope_prima, i) * avec_1(g, A_prima, Q_prima, Bw_prima, i)[0]
                flujos_2[i, 0] = lambda2_i * gamma2(g, n, Delta_x, A_prima, Q_prima,
                                                    Bw_prima, slope_prima, i) * avec_2(g, A_prima, Q_prima, Bw_prima, i)[0]
                flujos_1[i, 1] = lambda1_i * gamma1(g, n, Delta_x, A_prima, Q_prima,
                                                    Bw_prima, slope_prima, i) * avec_1(g, A_prima, Q_prima, Bw_prima, i)[1]
                flujos_2[i, 1] = lambda2_i * gamma2(g, n, Delta_x, A_prima, Q_prima,
                                                    Bw_prima, slope_prima, i) * avec_2(g, A_prima, Q_prima, Bw_prima, i)[1]
                flujos_1[i, 2] = np.sign(lambda1_i)
                flujos_2[i, 2] = np.sign(lambda2_i)

                q_right = flujo_numerico(
                    g, n, Delta_x, A_prima, Q_prima, Bw_prima, slope_prima, i)
                # q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
                flujos_num[i] += q_right*S[i]
                #flujos_num[i] += q_left*S[i-1] if q_left>0 else q_left*S[i]

            if i == 0:
                lambda1_i = lambda1(g, A, Q, Bw, i)
                lambda2_i = lambda2(g, A, Q, Bw, i)
                flujos_1[i, 0] = lambda1_i * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_1(g, A, Q, Bw, i)[0]
                flujos_2[i, 0] = lambda2_i * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_2(g, A, Q, Bw, i)[0]
                flujos_1[i, 1] = lambda1_i * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_1(g, A, Q, Bw, i)[1]
                flujos_2[i, 1] = lambda2_i * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_2(g, A, Q, Bw, i)[1]
                flujos_1[i, 2] = np.sign(lambda1_i)
                flujos_2[i, 2] = np.sign(lambda2_i)

                q_right = flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, i)
                flujos_num[i] += q_right * \
                    S[i] if q_right > 0 else q_right*S[i+1]

            if i > 0 and i < (nx_cell-1):
                lambda1_i = lambda1(g, A, Q, Bw, i)
                lambda2_i = lambda2(g, A, Q, Bw, i)
                flujos_1[i, 0] = lambda1_i * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_1(g, A, Q, Bw, i)[0]
                flujos_2[i, 0] = lambda2_i * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_2(g, A, Q, Bw, i)[0]
                flujos_1[i, 1] = lambda1_i * \
                    gamma1(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_1(g, A, Q, Bw, i)[1]
                flujos_2[i, 1] = lambda2_i * \
                    gamma2(g, n, Delta_x, A, Q, Bw, slope, i) * \
                    avec_2(g, A, Q, Bw, i)[1]
                flujos_1[i, 2] = np.sign(lambda1_i)
                flujos_2[i, 2] = np.sign(lambda2_i)

                q_right = flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, i)
                # q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
                flujos_num[i] += q_right * \
                    S[i] if q_right > 0 else q_right*S[i+1]
                #flujos_num[i] += q_left*S[i-1] if q_left>0 else q_left*S[i]

    return flujos_1, flujos_2, flujos_num


@jit
def update(g: float, n: float, A: np.ndarray, Q: np.ndarray, S: np.ndarray, Bw: np.ndarray, slope: np.ndarray, nx_cell, CFL, Delta_x, mode, up_contA: np.ndarray, up_contQ: np.ndarray, up_contS: np.ndarray, down_contA: np.ndarray, down_contQ: np.ndarray, t):

    Delta_t = calcula_dt(g, A, Q, Bw, nx_cell, CFL, Delta_x)
    
    A_pres = np.zeros(nx_cell)
    Q_pres = np.zeros(nx_cell)
    S_pres = np.zeros(nx_cell)
    flujos_1 = np.zeros((nx_cell, 3))
    flujos_2 = np.zeros((nx_cell, 3))

    flujos_num = np.zeros(nx_cell)
    coef = (Delta_t/Delta_x)

    flujos_1, flujos_2, flujos_num = lista_flujos(
        g, n, A, Q, S, Bw, slope, nx_cell, Delta_x)

    #if t % 1000 == 0:
        #print(Delta_t)

    for i in range(nx_cell):

        A_pres[i] = A[i]-coef*(flujos_1[i-1, 0]*(flujos_1[i-1, 2] == 1) +
                               flujos_2[i-1, 0]*(flujos_2[i-1, 2] == 1) +
                               flujos_1[i, 0]*(flujos_1[i, 2] == -1) +
                               flujos_2[i, 0]*(flujos_2[i, 2] == -1) +
                               flujos_1[i-1, 0]*(flujos_1[i-1, 2] == 0) +
                               flujos_2[i, 0]*(flujos_2[i, 2] == 0))

        Q_pres[i] = Q[i]-coef*(flujos_1[i-1, 1]*(flujos_1[i-1, 2] == 1)+
                                flujos_2[i-1, 1]*(flujos_2[i-1, 2] == 1) +
                                flujos_1[i, 1]*(flujos_1[i, 2] == -1) +
                                flujos_2[i, 1]*(flujos_2[i, 2] == -1) +
                               flujos_1[i-1, 1]*(flujos_1[i-1, 2] == 0) +
                               flujos_2[i, 1]*(flujos_2[i, 2] == 0))

        # soluto
        S_pres[i] = (A[i]*S[i]-coef*(flujos_num[i]-flujos_num[i-1]))/A_pres[i]

    if mode == 'free':
        pass

    if mode == 'supercritical':  # Supercritico
        A_pres[0] = up_contA
        Q_pres[0] = up_contQ

    if mode == 'subcritical':  # NO EXISTE->Subcritico
        Q_pres[0] = up_contQ
        A_pres[-1] = down_contA

    if mode == 'close':
        Q_pres[0] = 0.0
        Q_pres[-1] = 0.0

    if mode == 'both':
        A_pres[0] = up_contA
        Q_pres[0] = up_contQ
        Q_pres[-1] = down_contQ
        A_pres[-1] = down_contA

    S_pres[0] = up_contS

    return A_pres, Q_pres, S_pres, Delta_t

# @jit


def update2(g: float, n: float, A: np.ndarray, Q: np.ndarray, S: np.ndarray, Bw: np.ndarray, slope: np.ndarray, nx_cell, CFL, Delta_x, mode, up_contA: np.ndarray, up_contQ: np.ndarray, up_contS: np.ndarray, down_contA: np.ndarray, down_contQ: np.ndarray):

    Delta_t = calcula_dt(g, A, Q, Bw, nx_cell, CFL, Delta_x)
    
    A_pres = np.zeros(nx_cell)
    Q_pres = np.zeros(nx_cell)
    S_pres = np.zeros(nx_cell)
    coef = (Delta_t/Delta_x)

    for i in range(1, nx_cell-1):

        lambda1_p = 0.5*(lambda1(g, A, Q, Bw, i-1) +
                         abs(lambda1(g, A, Q, Bw, i-1)))
        lambda2_p = 0.5*(lambda2(g, A, Q, Bw, i-1) +
                         abs(lambda2(g, A, Q, Bw, i-1)))
        lambda1_m = 0.5*(lambda1(g, A, Q, Bw, i)-abs(lambda1(g, A, Q, Bw, i)))
        lambda2_m = 0.5*(lambda2(g, A, Q, Bw, i)-abs(lambda2(g, A, Q, Bw, i)))

        if lambda1_p != 0:
            A_pres[i] = A_pres[i] + lambda1_p * \
                gamma1(g, n, Delta_x, A, Q, Bw, slope, i-1) * \
                avec_1(g, A, Q, Bw, i-1)[0]
            Q_pres[i] = Q_pres[i] + lambda1_p * \
                gamma1(g, n, Delta_x, A, Q, Bw, slope, i-1) * \
                avec_1(g, A, Q, Bw, i-1)[1]

        if lambda2_p != 0:
            A_pres[i] = A_pres[i] + lambda2_p * \
                gamma2(g, n, Delta_x, A, Q, Bw, slope, i-1) * \
                avec_2(g, A, Q, Bw, i-1)[0]
            Q_pres[i] = Q_pres[i] + lambda2_p * \
                gamma2(g, n, Delta_x, A, Q, Bw, slope, i-1) * \
                avec_2(g, A, Q, Bw, i-1)[1]

        if lambda1_m != 0:
            A_pres[i] = A_pres[i] + lambda1_m * \
                gamma1(g, n, Delta_x, A, Q, Bw, slope, i) * \
                avec_1(g, A, Q, Bw, i)[0]
            Q_pres[i] = Q_pres[i] + lambda1_m * \
                gamma1(g, n, Delta_x, A, Q, Bw, slope, i) * \
                avec_1(g, A, Q, Bw, i)[1]

        if lambda2_m != 0:
            A_pres[i] = A_pres[i] + lambda2_m * \
                gamma2(g, n, Delta_x, A, Q, Bw, slope, i) * \
                avec_2(g, A, Q, Bw, i)[0]
            Q_pres[i] = Q_pres[i] + lambda2_m * \
                gamma2(g, n, Delta_x, A, Q, Bw, slope, i) * \
                avec_2(g, A, Q, Bw, i)[1]

        A_pres[i] = -coef*A_pres[i]
        Q_pres[i] = -coef*Q_pres[i]

        A_pres[i] = A_pres[i]+A[i]
        Q_pres[i] = Q_pres[i]+Q[i]

        # soluto
        ans = A[i]*S[i]
        q_right = flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, i)
        q_left = flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, i-1)
        ans -= coef*q_right*S[i] if q_right > 0 else coef*q_right*S[i+1]
        ans += coef*q_left*S[i-1] if q_left > 0 else coef*q_left*S[i]

        S_pres[i] = ans/A_pres[i]

        # A_pres[i]=0 if np.isnan(ans[0]) else ans[0]
        # Q_pres[i]=0 if np.isnan(ans[1]) else ans[1]

    lambda1_m = 0.5*(lambda1(g, A, Q, Bw, i)-abs(lambda1(g, A, Q, Bw, 0)))
    lambda2_m = 0.5*(lambda2(g, A, Q, Bw, i)-abs(lambda2(g, A, Q, Bw, 0)))
    lambda1_p = 0.5*(lambda1(g, A, Q, Bw, nx_cell-2) +
                     abs(lambda1(g, A, Q, Bw, nx_cell-2)))
    lambda2_p = 0.5*(lambda2(g, A, Q, Bw, nx_cell-2) +
                     abs(lambda2(g, A, Q, Bw, nx_cell-2)))

    if mode == 'free':
        A_pres[0] = A_pres[1]
        A_pres[-1] = A_pres[-2]
        Q_pres[0] = Q_pres[1]
        Q_pres[-1] = Q_pres[-2]
        # A_pres[-1] = A[-1] + coef*(lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_1(g,A,Q,Bw,-2)[0]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_2(g,A,Q,Bw,-2)[0])
        # Q_pres[-1] = Q[-1] + coef*(lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_1(g,A,Q,Bw,-2)[1]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_2(g,A,Q,Bw,-2)[1])
        # A_pres[0] = A[0] + coef*(lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,0)*avec_1(g,A,Q,Bw,0)[0]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,0)*avec_2(g,A,Q,Bw,0)[0])
        # Q_pres[0] = Q[0] + coef*(lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,0)*avec_1(g,A,Q,Bw,0)[1]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,0)*avec_2(g,A,Q,Bw,0)[1])

    if mode == 'supercritical':  # Supercritico
        A_pres[0] = up_contA
        Q_pres[0] = up_contQ
        A_pres[-1] = A_pres[-2]
        Q_pres[-1] = Q_pres[-2]

    if mode == 'subcritical':  # NO EXISTE->Subcritico
        A_pres[0] = A_pres[1]
        Q_pres[0] = up_contQ
        A_pres[-1] = down_contA
        Q_pres[-1] = Q_pres[-2]

    if mode == 'both':
        A_pres[0] = up_contA
        Q_pres[0] = up_contQ
        A_pres[-1] = down_contA
        Q_pres[-1] = down_contQ

    if mode == 'close':
        A_pres[0] = A_pres[1]
        A_pres[-1] = A_pres[-2]
        Q_pres[0] = 0.0
        Q_pres[-1] = 0.0

    S_pres[0] = up_contS
    S_pres[-1] = S_pres[-2]
    # A_prima
    # ans = A[-1]*S[-1]
    #   q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i)
    #   q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
    #   ans -= coef*q_right*S[i] if q_right>0 else coef*q_right*S[i+1]
    #   ans += coef*q_left*S[i-1] if q_left>0 else coef*q_left*S[i]

    #   S_pres[i] = ans/A_pres[i]

    return A_pres, Q_pres, S_pres

# PLOTS


def plot_perfil_soluto(A, Q, S, x, Base_width, slope_z, t, t_real, rangeA=[0, 0], rangeQ=[0, 0], dir='./images/soluto', mc=0):
    fig2, ax2 = plt.subplots()
    color = 'tab:blue'
    ax2.set_title(f'Perfil de $U=(A,Q,\Phi)$; t={t_real:.2f} s')
    ax2.set_xlabel(f'$x [m]$')
    ax2.set_ylabel('$h[m]$', color=color)
    # ax2.grid()
    ax2.plot(x, A/Base_width+slope_z, '-', color=color)
    ax2.plot(x, S, '.', ms=2.5, color='green', label='$\Phi(x)$')
    ax2.set_ylim(rangeA)
    # ax2.set_xlim([0,length])
    ax2.tick_params(axis='y', labelcolor=color)
    if sum(slope_z) != 0:
        ax2.fill_between(x, slope_z, 0, color='grey')

    ax22 = ax2.twinx()
    color = 'tab:red'
    ax22.set_ylabel('$Q[m^3/s]$', color=color)
    ax22.grid()
    ax22.plot(x, Q, '-', color=color)
    ax22.set_ylim(rangeQ)
    # ax22.set_xlim([0,length])
    ax22.tick_params(axis='y', labelcolor=color)

    if mc > 0:
        xM, QM, hM, hzM, bM = np.array(import_MC(mc))
        ax2.plot(xM[::5], hzM[::5], '.', ms=2, color='cyan', label='McDonald')

    fig2.tight_layout()
    ax2.legend()
    quality = 250

    fig2.savefig(dir+f'{t}.jpg', transparent=False,
                 dpi=quality)  # , facecolor='w')


def gifiyer(path, nt_cell, *, freeze=0, paso=1, FPS=10, gif_soluto=''):
    if len(gif_soluto) < 2:
        gif_soluto = 'soluto_animation_mc.gif'
    images_data2 = []

    for i in range(freeze, 1):
        data2 = imageio.imread(path+f'{i}.jpg')
        images_data2.append(data2)

    for i in range(1, nt_cell-1):
        if i % paso == 0:
            data2 = imageio.imread(path+f'{i+1}.jpg')
            images_data2.append(data2)

    imageio.mimwrite(gif_soluto, images_data2, format='.gif', fps=FPS)

# Para busqueda del estacionario


@jit
def estabilidad(old, new):
    resta = old-new
    return abs(sum(resta/old))

# Para el adjunto
# @jit


def soluto_forward_plot(g, n, k, A, Q, S_inicio, Bw, slope, nx_cell, nt_cell, Delta_x, Delta_t, up_contS, freq=0, *, x_ax, Arange, Qrange):
    S_old = S_inicio.copy()
    S_new = S_inicio.copy()

    S_cont = np.zeros(nt_cell)
    S_cont[0] = S_new[-1]
    for t in np.arange(nt_cell):

        coef = (Delta_t/Delta_x)
        S_new = np.zeros(nx_cell)
        flujos_num = np.zeros(nx_cell)
        coef = (Delta_t/Delta_x)

        null, null, flujos_num = lista_flujos(
            g, n, A, Q, S_old, Bw, slope, nx_cell, Delta_x)
        for x in range(nx_cell):
            S_new[x] = (A[x]*S_old[x]-coef*(flujos_num[x] -
                        flujos_num[x-1]))/A[x]-k*S_old[x]

        S_new[0] = up_contS[t]
        # S_new[-1]=S_new[-2]
        S_old = S_new.copy()
        S_cont[t] = S_new[-1]

        if freq != 0 and t % freq == 0:
            plot_perfil_soluto2(A, Q, S_new, x_ax, Bw, slope,
                                t+1, t*Delta_t, Arange, Qrange)
            plt.close('all')

    return S_new, S_cont, nt_cell

# @njit
# def soluto_forward(g,n,k,A,Q,S_inicio,Bw,slope,nx_cell,nt_cell,Delta_x,Delta_t,up_contS):
#   S_old = S_inicio.copy()
#   S_new = S_inicio.copy()

#   S_cont= np.zeros(nt_cell)
#   S_cont[0]=S_new[-1]
#   for t in np.arange(nt_cell):

#     coef = (Delta_t/Delta_x)
#     S_new=np.zeros(nx_cell)
#     flujos_num=np.zeros(nx_cell)
#     coef = (Delta_t/Delta_x)

#     null,null,flujos_num = lista_flujos(g,n,A,Q,S_old,Bw,slope,nx_cell,Delta_x)
#     for x in range(nx_cell):
#       S_new[x] = (A[x]*S_old[x]-coef*(flujos_num[x]-flujos_num[x-1]))/A[x]-k*S_old[x]

#     S_new[0]=up_contS[t]
#     #S_new[-1]=S_new[-2]
#     S_old=S_new.copy()
#     S_cont[t]=S_new[-1]

#   return S_new,S_cont,nt_cell


@jit
def soluto_forward(g, n, k, A, Q, S_inicio, Bw, slope, nx_cell, nt_cell, Delta_x, Delta_t, up_contS):
    S_old = S_inicio.copy()
    S_new = S_inicio.copy()

    S_cont = np.zeros(nt_cell)
    # S_cont[0]=S_new[-1]
    flujos_num = np.zeros(nx_cell)

    for x in range(nx_cell):
        if x == nx_cell-1:

            A_prima = np.append(A, A[-1])
            Q_prima = np.append(Q, Q[-1])
            Bw_prima = np.append(Bw, Bw[-1])
            slope_prima = np.append(slope, slope[-1])
            q_right = flujo_numerico(
                g, n, Delta_x, A_prima, Q_prima, Bw_prima, slope_prima, x)
            # q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
            flujos_num[x] += q_right
        if x == 0:
            q_right = flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, x)
            flujos_num[x] = q_right
        if x > 0 and x < (nx_cell-1):

            q_right = flujo_numerico(g, n, Delta_x, A, Q, Bw, slope, x)
            # q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
            flujos_num[x] += q_right

    coef = (Delta_t/Delta_x)

    for t in np.arange(nt_cell):

        for x in range(1, nx_cell):
            ans = S_old[x]*A[x]
            if x == nx_cell-1:
                ans -= coef*flujos_num[x]*S_old[x]
                ans += coef * \
                    flujos_num[x-1]*S_old[x-1] if flujos_num[x -
                                                             1] > 0 else coef*flujos_num[x-1]*S_old[x]

                #S_new[x] = ans/A[x]

            if x > 0 and x < nx_cell-1:
                ans -= coef * \
                    flujos_num[x]*S_old[x] if flujos_num[x] > 0 else coef * \
                    flujos_num[x]*S_old[x+1]
                ans += coef * \
                    flujos_num[x-1]*S_old[x-1] if flujos_num[x -
                                                             1] > 0 else coef*flujos_num[x-1]*S_old[x]

            S_new[x] = ans/A[x]-k*S_old[x]

        S_new[0] = up_contS[t]

        S_old = S_new.copy()
        S_cont[t] = S_new[-1]

    return S_new, S_cont

# Función objetivo


def import_river():
    directorio = "./river/shallow.txt"
    x, A, Q, Slope, b = zip(*(map(float, line.split())
                            for line in open(directorio, 'r')))
    return x, A, Q, Slope, b


@jit
def objetivo(simulacion, _medidas):
    Delta = (simulacion - _medidas)**2
    return 1/(2*len(Delta))*np.sum(Delta)

# evolucion adjunta solo hasta xm


@jit
def evolucion_inversa(soluto: np.ndarray, _medidas: np.ndarray, nx, nt, Dx, Dt, Q, A, K):
    times = (range(nt-1))
    Delta_x = Dx
    sigma_prev = np.zeros(nx)
    sigma_new = np.zeros(nx)
    sigma_contorno = np.zeros(nt)
    for tinv in times:
        t = nt-2-tinv
        Delta_t = Dt
        sigma_prev = sigma_new.copy()
        for x in range(nx-1):

            Q_ = Q[x]
            A_ = A[x]
            u_ = Q_/A_
            Dsig = sigma_prev[x+1]-sigma_prev[x]
            #Dsig_minus = _fases_sig[x,t+1]-_fases_sig[x-1,t+1]

            # +  uplus * Dphi_plus) ??? Cambia el sentido de u
            sigma_new[x] = sigma_prev[x] + Delta_t / Delta_x * (u_ * Dsig)

            # if x==0:
            #   _fases_sig[x,t] += E*Delta_t/(Delta_x**2) * (_fases_sig[x+2,t+1]-2*_fases_sig[x+1,t+1]+_fases_sig[x,t+1])
            # else:
            #   _fases_sig[x,t] += E*Delta_t/(Delta_x**2) * (_fases_sig[x+1,t+1]-2*_fases_sig[x,t+1]+_fases_sig[x-1,t+1])

            if (x+1) == (nx-1):
                aux = -(soluto[t+1]-_medidas[t+1]) * Delta_t / A_
                # print(aux)
                sigma_new[x] = sigma_new[x] + aux

            sigma_new[x] = sigma_new[x] + sigma_prev[x] * K * Delta_t

            #if _fases_sig[x,t]<0: _fases_sig[x,t] *= 0
        sigma_contorno[t] = sigma_new[0]

    return sigma_new, sigma_contorno


# Evolucion del contorno OOOOJOO QUE ESTOY COMPARANDO MEDIDAS Y AUX
@jit
def nuevo_cont_eps_i(g, n, k, A, Q, Bw, slope, nx_cell, nt_cell, Delta_x, Dt_list, up_contS, previo, gradiente, medidas, eps):

    aux = previo + gradiente*eps
    S_inicio = aux.copy()

    _, contorno = soluto_forward(
        g, n, k, A, Q, S_inicio, Bw, slope, nx_cell, nt_cell, Delta_x, Dt_list, up_contS)
    return objetivo(contorno, medidas)


@jit
def nuevo_cont_eps_c(g, n, k, A, Q, Bw, slope, nx_cell, nt_cell, Delta_x, Dt_list, previo, S_ini, gradiente, medidas, eps):

    aux = previo + gradiente*eps
    up_contS = aux.copy()

    _, contorno = soluto_forward(
        g, n, k, A, Q, S_ini, Bw, slope, nx_cell, nt_cell, Delta_x, Dt_list, up_contS)
    return objetivo(contorno, medidas)


# McDOnald Cases

def import_MC(caso):
    directorio = "/home/sahara-rebel/Desktop/TFG/SIMULACIONES/macdonaldTestcases1-6/"
    x, Q, h, hz, _, _, b, *_ = zip(*(map(float, line.split())
                                   for line in open(directorio + f'{caso}-00.sol', 'r')))
    return x, Q, h, hz, b


def plot_with_mc(caso, rangeA=[0, 0], rangeQ=[0, 0], dir='./images/mcdonald/'):

    x, Q, h, hz, b = np.array(import_MC(caso))
    slope_z = hz-h
    print('length:', len(x))

    fig2, ax2 = plt.subplots()
    color = 'cyan'
    ax2.set_title(f'Perfil de $U=(A,Q,\Phi)$; estacionario McDonald')
    ax2.set_xlabel(f'$x [m]$')
    ax2.set_ylabel('$h[m]$', color=color)
    # ax2.grid()
    ax2.plot(x, h+slope_z, '-', color=color)

    ax2.set_ylim(rangeA)
    # ax2.set_xlim([0,length])
    ax2.tick_params(axis='y', labelcolor=color)
    if sum(slope_z) != 0:
        ax2.fill_between(x, slope_z, 0, color='grey')

    ax22 = ax2.twinx()
    color = 'tab:red'
    ax22.set_ylabel('$Q[m^3/s]$', color=color)
    # ax22.grid()
    ax22.plot(x, Q, '-', color=color)
    ax22.set_ylim(rangeQ)
    # ax22.set_xlim([0,length])
    ax22.tick_params(axis='y', labelcolor=color)

    fig2.tight_layout()

    quality = 90

    # , facecolor='w')
    fig2.savefig(dir+f'mcdonald{caso}.jpg', transparent=False, dpi=quality)
    # plt.show()


def plot_perfil_soluto2(A, Q, S, x, Base_width, slope_z, t, t_real, rangeA=[0, 0], rangeQ=[0, 0], dir='./images/soluto/'):
    fig2, ax2 = plt.subplots()
    color = 'tab:blue'
    ax2.set_title(f'Perfil de $U=(A,Q,\Phi)$; t={t_real:.2f} s')
    ax2.set_xlabel(f'$x [m]$')
    ax2.set_ylabel('$h[m]$', color=color)
    # ax2.grid()
    ax2.plot(x, A/Base_width+slope_z, '-', color=color)
    ax2.plot(x, S+slope_z, '.', ms=2.5, color='green', label='$\Phi(x)$')
    ax2.set_ylim(rangeA)
    # ax2.set_xlim([0,length])
    ax2.tick_params(axis='y', labelcolor=color)
    if sum(slope_z) != 0:
        ax2.fill_between(x, slope_z, 0, color='grey')

    ax22 = ax2.twinx()
    color = 'tab:red'
    ax22.set_ylabel('$Q[m^3/s]$', color=color)
    # ax22.grid()
    ax22.plot(x, Q, '-', color=color)
    ax22.set_ylim(rangeQ)
    # ax22.set_xlim([0,length])
    ax22.tick_params(axis='y', labelcolor=color)

    fig2.tight_layout()
    ax2.legend()
    quality = 150

    fig2.savefig(dir+f'{t}.jpg', transparent=False,
                 dpi=quality)  # , facecolor='w')




###################################### CONFLUENCIAS ##################################################



def inicializa(N_canales,file):
    _L=np.zeros(N_canales)
    _manning=np.zeros(N_canales)
    _mode=[[]]*N_canales
    _nx_cell=[[]]*N_canales

    with open(file) as f:
        for i,line in enumerate(f):
            if i>0:
                _L[i-1],_nx_cell[i-1],_mode[i-1],_manning[i-1],*_=line.split()
                _nx_cell[i-1]=int(_nx_cell[i-1])

    return _L,_nx_cell,_mode,_manning
            

def carga_red(N_canales,file_path):
    _L=np.zeros(N_canales)
    _manning=np.zeros(N_canales)
    _mode=[[]]*N_canales
    _nx_cell=[[]]*N_canales
    _matrix=np.zeros((N_canales,N_canales))
    _qup=np.zeros(N_canales)
    _aup=np.zeros(N_canales)
    _qdown=np.zeros(N_canales)
    _adown=np.zeros(N_canales)

    with open(file_path+'Canales.txt') as f:
        for i,line in enumerate(f):
            if i>0:
                _L[i-1],_nx_cell[i-1],_mode[i-1],_manning[i-1],_qup[i-1],_aup[i-1],_qdown[i-1],_adown[i-1]=line.split()
                _nx_cell[i-1]=int(_nx_cell[i-1])

    with open(file_path+'matriz-red.txt') as g:
        for i,line in enumerate(g):
            _matrix[i]=line.split()
            

    Base_width = [[]]*N_canales
    Slope_z = [[]]*N_canales
    A_0 = [[]]*N_canales
    Q_0 = [[]]*N_canales

    for i in range(N_canales):
        Base_width[i] , Slope_z[i], A_0[i],Q_0[i]= zip(*(map(float, line.split())
                                   for line in open(file_path+ f'canal{i}.txt', 'r')))


    return _L,_nx_cell,_mode,_manning,np.array(Base_width),np.array(Slope_z),np.array(A_0),np.array(Q_0), _matrix,_aup,_qup,_adown,_qdown
    

@jit
def lista_flujos_red(g: float, n,
                    A, Q,
                    Bw, slope,
                    nx_cell, Delta_x):

                   
  flujos_right=np.zeros((nx_cell,2))
  flujos_left=np.zeros((nx_cell,2))
  flujos_num=np.zeros(nx_cell)#Q.copy()
  for i in range(nx_cell):
    lambda1_p = 0.5*(lambda1(g,A,Q,Bw,i-1)+abs(lambda1(g,A,Q,Bw,i-1)))
    lambda2_p = 0.5*(lambda2(g,A,Q,Bw,i-1)+abs(lambda2(g,A,Q,Bw,i-1)))
    lambda1_m = 0.5*(lambda1(g,A,Q,Bw,i)-abs(lambda1(g,A,Q,Bw,i)))
    lambda2_m = 0.5*(lambda2(g,A,Q,Bw,i)-abs(lambda2(g,A,Q,Bw,i)))

    if i==nx_cell-1:
      flujos_left[i,0] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[0]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[0]
      flujos_left[i,1] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[1]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[1]
      
    if i==0:
      flujos_right[i,0] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[0]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[0]
      flujos_right[i,1] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[1]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[1]
      
    if i>0 and i<(nx_cell-1):
      flujos_left[i,0] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[0]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[0]
      flujos_left[i,1] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[1]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[1]
      
      flujos_right[i,0] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[0]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[0]
      flujos_right[i,1] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[1]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[1]

  return flujos_left,flujos_right

def calcula_in_out(_matrix,A,Q,index):
    up_a=[]
    up_q=[]
    down_a=[]
    down_q=[]
    return up_a,up_q,down_a,down_q


#ahora A,Q serán matrices
#@njit
def update_red(g: float, n_red, A_red, Q_red, Bw_red, slope_red, 
                nx_cell_red, N_C, CFL, Delta_x_red, mode_red,
                up_contA_red, up_contQ_red, down_contA_red, down_contQ_red,
                matrix, t, t_r):

    A_new=[[]]*N_C
    Q_new=[[]]*N_C
    

    Delta_t=min([calcula_dt(g, np.array(A_red[j]), np.array(Q_red[j]), Bw_red[j], nx_cell_red[j], CFL, Delta_x_red[j]) for j in range(N_C)])
    #print (Delta_t)
    # Dt=np.zeros(N_C)
    # for i in range(N_C):
    #     Dt[i]=calcula_dt(g, np.array(A_red[i]), np.array(Q_red[i]), Bw_red[i], nx_cell_red[i], CFL, Delta_x_red[i])

    # Delta_t=min(Dt)

    if t_r+Delta_t>t:
        Delta_t=t-t_r

    for j in range(N_C):

        A=np.array(A_red[j])
        Q=np.array(Q_red[j])
        Bw=np.array(Bw_red[j])
        n=n_red[j]
        slope=np.array(slope_red[j])
        nx_cell=nx_cell_red[j]
        Delta_x=Delta_x_red[j]
        mode=mode_red[j]

        
        A_pres=np.zeros(nx_cell)
        Q_pres=np.zeros(nx_cell)

        flujos_right=np.zeros((nx_cell,2))
        flujos_left=np.zeros((nx_cell,2))

        coef = (Delta_t/Delta_x)

        flujos_left, flujos_right = lista_flujos_red(
            g, n, A, Q, Bw, slope, nx_cell, Delta_x,)

        #if t % 1000 == 0:
            #print(Delta_t)

        for i in range(nx_cell):

            A_pres[i] = A[i]-coef*(flujos_left[i,0]+flujos_right[i,0])
            Q_pres[i] = Q[i]-coef*(flujos_left[i,1]+flujos_right[i,1])

        A_new[j]=A_pres
        Q_new[j]=Q_pres
        
    for j in range(N_C):
        #Calculamos los contornos
        up_contA=np.array(up_contA_red[j])
        up_contQ=np.array(up_contQ_red[j])
        down_contA=np.array(down_contA_red[j])
        down_contQ=np.array(down_contQ_red[j])

        if sum(matrix[:,j])>0:
            up_contQ=0
            for index in range(N_C):
                if matrix[index,j]:
                    up_contQ+=Q_new[index][-1]

        if sum(matrix[j,:])>0:
            down_contA=0
            for index in range(N_C):
                if matrix[j,index]:
                    down_contA+=A_new[index][0]
            down_contA/=sum(matrix[j,:])

        if mode == 'free':
            pass

        if mode == 'supercritical':  # Supercritico
            A_new[j][0] = up_contA
            Q_new[j][0] = up_contQ

        if mode == 'subcritical':  # NO EXISTE->Subcritico
            Q_new[j][0] = up_contQ
            A_new[j][-1] = down_contA

        if mode == 'close':
            Q_new[j][0] = 0.0
            Q_new[j][-1] = 0.0

        if mode == 'both':
            A_new[j][0] = up_contA
            Q_new[j][0] = up_contQ
            Q_new[j][-1] = down_contQ
            A_new[j][-1] = down_contA

     
    

    return A_new, Q_new, Delta_t




def plot_perfil_soluto_red(A_red, Q_red, x_red, Bw_red, slope_red, t, t_real,N_canal, rangeA=[0, 0], rangeQ=[0, 0], dir='./images/shallow'):
    plt.style.use('Solarize_Light2')
    for i in range(N_canal):

        A=np.array(A_red[i])
        Q=np.array(Q_red[i])
        Bw=np.array(Bw_red[i])
        slope=np.array(slope_red[i])
        x=np.array(x_red[i])

        fig2, ax2 = plt.subplots()
        color = 'tab:blue'
        ax2.set_title(f'Perfil de $U=(A,Q)$; t={t_real:.2f} s')
        ax2.set_xlabel(f'$x [m]$')
        ax2.set_ylabel('$h[m]$', color=color)
        # ax2.grid()
        ax2.plot(x, A/Bw+slope, '-', color=color)
        ax2.set_ylim(rangeA)
        # ax2.set_xlim([0,length])
        ax2.tick_params(axis='y', labelcolor=color)
        if sum(slope) != 0:
            ax2.fill_between(x, slope, 0, color='grey')

        ax22 = ax2.twinx()
        color = 'tab:red'
        ax22.set_ylabel('$Q[m^3/s]$', color=color)
        ax22.grid()
        ax22.plot(x, Q, '-', color=color)
        ax22.set_ylim(rangeQ)
        # ax22.set_xlim([0,length])
        ax22.tick_params(axis='y', labelcolor=color)


        fig2.tight_layout()
        #ax2.legend()
        quality = 150

        fig2.savefig(dir+f'canal{i}_t{t}.jpg', transparent=False,
                    dpi=quality)  # , facecolor='w')


def gifiyer_red(path, nt_cell, N_c,*, freeze=0, paso=1, FPS=10):
    for j in range(N_c):
        
        gif_soluto = f'soluto_animation_shallow_canal_{j}.gif'
        images_data2 = []


        for i in range(1, nt_cell-1):
            if i % paso == 0:
                data2 = imageio.imread(path+f'canal{j}_t{i+1}.jpg')
                images_data2.append(data2)

        imageio.mimwrite(gif_soluto, images_data2, format='.gif', fps=FPS)




def plot_all_in_one(A_red, Q_red, x_red, Bw_red, slope_red, t, t_real,N_canal, rangeA=[0, 0], rangeQ=[0, 0], dir='./images/shallow'):
    plt.style.use('Solarize_Light2')
    fig2, ax2 = plt.subplots(N_canal,figsize=(10,10))
    ax2[0].set_title(f'Perfil de $U=(A,Q)$; t={t_real:.2f} s')
    for i in range(N_canal):

        A=np.array(A_red[i])
        Q=np.array(Q_red[i])
        Bw=np.array(Bw_red[i])
        slope=np.array(slope_red[i])
        x=np.array(x_red[i])

        
        color = 'tab:blue'
        #ax2[i].set_title(f'Perfil de $U=(A,Q)$; t={t_real:.2f} s')
        
        ax2[i].set_ylabel(f'$h_{i}[m]$', color=color)
        # ax2[i].grid()
        ax2[i].plot(x, A/Bw+slope, '-', color=color)
        ax2[i].set_ylim(rangeA)
        # ax2[i].set_xlim([0,length])
        ax2[i].tick_params(axis='y', labelcolor=color)
        if sum(slope) != 0:
            ax2[i].fill_between(x, slope, 0, color='grey')

        ax22 = ax2[i].twinx()
        color = 'tab:red'
        ax22.set_ylabel(f'$Q_{i}[m^3/s]$', color=color)
        ax22.grid()
        ax22.plot(x, Q, '-', color=color)
        ax22.set_ylim(rangeQ)
        # ax22.set_xlim([0,length])
        ax22.tick_params(axis='y', labelcolor=color)

    ax2[-1].set_xlabel(f'$x [m]$')

    fig2.tight_layout()
    #ax2.legend()
    quality = 130

    fig2.savefig(dir+f'{N_canal}-canales_t{t}.jpg', transparent=False,
                dpi=quality)  # , facecolor='w')


def gifiyer_all_in_one(path, nt_cell, *, freeze=0, paso=1, FPS=10,gif_soluto = 'shallow_all_canals.gif'):
    
    
    images_data2 = []

    for i in range(1, nt_cell-1):
        if i % paso == 0:
            data2 = imageio.imread(path+f'{i}.jpg')
            images_data2.append(data2)
    
    data2 = imageio.imread(path+f'{nt_cell}.jpg')
    images_data2.append(data2)

    imageio.mimwrite(gif_soluto, images_data2, format='.gif', fps=FPS)

def guarda_estacionario(Bw,Sz,A,Q,N_c,path,*,mode='shallow',S=0.0):
    for i in range(N_c):
        if mode=='shallow':
            open(path+f'canal{i}.txt', 'w').writelines(list('\t'.join(map(str, med_set)) +
                                        '\n' for med_set in zip(Bw[i],Sz[i],A[i],Q[i])))
        if mode=='soluto':
            open(path+f'canal{i}.txt', 'w').writelines(list('\t'.join(map(str, med_set)) +
                                        '\n' for med_set in zip(Bw[i],Sz[i],A[i],Q[i],S[i])))


########### SOLUTO-RED ##########

def plot_all_in_one_ws(A_red, Q_red, S_red, x_red, Bw_red, slope_red, t, t_real,N_canal, rangeA=[0, 0], rangeQ=[0, 0], dir='./images/Soluto'):
    plt.style.use('Solarize_Light2')
    fig2, ax2 = plt.subplots(N_canal,figsize=(10,10))
    ax2[0].set_title(f'Perfil de $U=(A,Q)$; t={t_real:.2f} s')
    for i in range(N_canal):

        A=np.array(A_red[i])
        Q=np.array(Q_red[i])
        S=np.array(S_red[i])
        Bw=np.array(Bw_red[i])
        slope=np.array(slope_red[i])
        x=np.array(x_red[i])

        
        color = 'tab:blue'
        #ax2[i].set_title(f'Perfil de $U=(A,Q)$; t={t_real:.2f} s')
        
        ax2[i].set_ylabel(f'$h_{i}[m]$', color=color)
        # ax2[i].grid()
        ax2[i].plot(x, A/Bw+slope, '-', color=color)
        ax2[i].plot(x, S, '.', ms=2.5, color='green', label='$\Phi(x)$')
        ax2[i].set_ylim(rangeA)
        # ax2[i].set_xlim([0,length])
        ax2[i].tick_params(axis='y', labelcolor=color)
        if sum(slope) != 0:
            ax2[i].fill_between(x, slope, 0, color='grey')

        ax22 = ax2[i].twinx()
        color = 'tab:red'
        ax22.set_ylabel(f'$Q_{i}[m^3/s]$', color=color)
        ax22.grid()
        ax22.plot(x, Q, '-', color=color)
        ax22.set_ylim(rangeQ)
        # ax22.set_xlim([0,length])
        ax22.tick_params(axis='y', labelcolor=color)

    ax2[-1].set_xlabel(f'$x [m]$')

    fig2.tight_layout()
    #ax2.legend()
    quality = 130

    fig2.savefig(dir+f'{N_canal}-canales_t{t}.jpg', transparent=False,
                dpi=quality)  # , facecolor='w')


@jit
def nuevo_cont_eps_red(g, n, k, A, Q, Bw, slope, nx_cell, nt_cell, Delta_x, Dt_list, previo, S_ini, gradiente, medidas, eps):

    aux = previo + gradiente*eps
    up_contS = aux.copy()

    _, contorno = soluto_forward_red(
        g, n, k, A, Q, S_ini, Bw, slope, nx_cell, nt_cell, Delta_x, Dt_list, up_contS)
    return objetivo(contorno, medidas)


#@jit
def soluto_forward_red(g, n, k, E, A, Q, S_inicio, Bw, slope, x_axis,nx_cell, nt_cell, Delta_x, Delta_t, up_contS,N_c,matrix,plot=False, fr=10):
    
    flujos_num = [np.zeros(nx_cell[i]) for i in range(N_c)]
    
    S_old = S_inicio.copy()
    S_new = S_inicio.copy()
    S_cont = np.zeros((N_c,nt_cell))
        # S_cont[0]=S_new[-1]
    for i in range(N_c):
        

        for x in range(nx_cell[i]):
            if x == nx_cell[i]-1:

                # A_prima = np.append(A[i], A[i][-1])
                # Q_prima = np.append(Q[i], Q[i][-1])
                # Bw_prima = np.append(Bw[i], Bw[i][-1])
                # slope_prima = np.append(slope[i], slope[i][-1])
                # q_right = flujo_numerico(
                #     g, n[i], Delta_x[i], A_prima, Q_prima, Bw_prima, slope_prima, x)
                q_right = flujo_numerico(g, n[i], Delta_x[i], A[i], Q[i], Bw[i], slope[i], x-1)
                flujos_num[i][x] = q_right
            if x == 0:
                q_right = flujo_numerico(g, n[i], Delta_x[i], A[i], Q[i], Bw[i], slope[i], x)
                flujos_num[i][x] = q_right
            if x > 0 and x < (nx_cell[i]-1):

                q_right = flujo_numerico(g, n[i], Delta_x[i], A[i], Q[i], Bw[i], slope[i], x)
                # q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
                flujos_num[i][x] = q_right

        coef = (Delta_t/Delta_x[i])

    for t in np.arange(nt_cell):
        for i in range(N_c):
            coef = (Delta_t/Delta_x[i])
            for x in range(1, nx_cell[i]):
                ans = S_old[i][x]*A[i][x]
                if x == nx_cell[i]-1:
                    ans -= coef*flujos_num[i][x]*S_old[i][x]
                    ans += coef * \
                        flujos_num[i][x-1]*S_old[i][x-1] if flujos_num[i][x -
                                                                1] > 0 else coef*flujos_num[i][x-1]*S_old[i][x]

                    #S_new[i][x] = ans/A[i][x]

                if x > 0 and x < nx_cell[i]-1:
                    ans -= coef * \
                        flujos_num[i][x]*S_old[i][x] if flujos_num[i][x] > 0 else coef * \
                        flujos_num[i][x]*S_old[i][x+1]
                    ans += coef * \
                        flujos_num[i][x-1]*S_old[i][x-1] if flujos_num[i][x -
                                                                1] > 0 else coef*flujos_num[i][x-1]*S_old[i][x]


                S_new[i][x] = ans/A[i][x]-k*S_old[i][x] 
                if x==nx_cell[i]-1:
                    S_new[i][x] += E*Delta_t/(Delta_x[i]**2) * (S_old[i][x]-2*S_old[i][x-1]+S_old[i][x-2])
                else:
                    S_new[i][x] += E*Delta_t/(Delta_x[i]**2) * (S_old[i][x+1]-2*S_old[i][x]+S_old[i][x-1])


        for i in range(N_c):
            # COntorno
            aux=up_contS[i][t]
            if sum(matrix[:,i])>0:
                aux=0
                for index in range(N_c):
                    if matrix[index,i]:
                        aux+=Q[index][-1]*S_new[index][-1]
                aux/=Q[i][0]


            S_new[i][0] = aux

            S_old[i] = S_new[i].copy()
            S_cont[i][t] = S_new[i][-1]

        if plot and t%fr==0:
            plot_all_in_one_ws(A,Q,S_new,x_axis,Bw,slope,t,t*Delta_t,N_c,[0,5],[0,8],'./images/Soluto/')
            plt.close('all')

    return S_new, S_cont


# La parte adjunta esta desarrollada específicamente para el llamado caso T

def evolucion_inversa_red(soluto, _medidas, nx_cell, nt, Dx, Dt, Q, A, K, E, N_c, matrix):

    sigma=np.zeros(nt)
    sigma_prev = [np.zeros(nx_cell[i]) for i in range(N_c)]
    sigma_new = [np.zeros(nx_cell[i]) for i in range(N_c)]


    for t in np.flip(range(nt)):

        for i in range(N_c):

            Delta_x=Dx[i]
            nx=nx_cell[i]

            for x in range(nx-1):

                Q_ = Q[i][x]
                A_ = A[i][x]
                u_ = Q_/A_
                Dsig = sigma_prev[i][x+1]-sigma_prev[i][x]
                #Dsig_minus = _fases_sig[i][x,t+1]-_fases_sig[i][x-1,t+1]

                # +  uplus * Dphi_plus) ??? Cambia el sentido de u
                sigma_new[i][x] = sigma_prev[i][x] + Dt / Delta_x * (u_ * Dsig)

                # if x==0:
                #   _fases_sig[i][x,t] += E*Delta_t/(Delta_x**2) * (_fases_sig[i][x+2,t+1]-2*_fases_sig[i][x+1,t+1]+_fases_sig[i][x,t+1])
                # else:
                #   _fases_sig[i][x,t] += E*Delta_t/(Delta_x**2) * (_fases_sig[i][x+1,t+1]-2*_fases_sig[i][x,t+1]+_fases_sig[i][x-1,t+1])

                if (x+1) == (nx-1):
                    if i==0:
                        aux = -(soluto[t]-_medidas[t]) * Dt / A_
                        # print(aux)
                        sigma_new[i][x] = sigma_new[i][x] + aux
                    else:
                        
                        prop=1*(i==1)+0*(i==2)#(Q[i][-1]/Q[0][0])
                        sigma_new[i][-1]=sigma_prev[0][0]*prop


                sigma_new[i][x] = sigma_new[i][x] + sigma_prev[i][x] * K * Dt 
                if (x==0):
                    sigma_new[i][x] += E*Dt/(Delta_x**2) * (sigma_prev[i][x+2]-2*sigma_prev[i][x+1]+sigma_prev[i][x])
                else:
                    sigma_new[i][x] += E*Dt/(Delta_x**2) * (sigma_prev[i][x+1]-2*sigma_prev[i][x]+sigma_prev[i][x-1])
                               

        sigma_prev=sigma_new.copy()
        sigma[t]=sigma_new[1][0]


    return sigma