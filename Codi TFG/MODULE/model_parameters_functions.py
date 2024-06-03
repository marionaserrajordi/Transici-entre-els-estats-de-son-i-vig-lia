import numpy as np

# Llavors per a la generació de nombres aleatoris
entropy_seed = 12345

# Paràmetres de la simulació
T = 10  # temps de simulació en segons
dt = 0.1  # pas de temps en ms
n = int(T * 1000 / dt)  # nombre de passos de temps
n_initial = int(4000 / dt)  # passos de temps inicials
sens_duration = 100  # durada de l'entrada sensorial
sens_input_iter = int(sens_duration / dt)  # iteracions per a l'entrada sensorial
n_1 = int(n / 2)  # temps d'implementació sensorial

# Paràmetres de l'escalat sinàptic
beta_ampa_intra = 1
beta_ampa_input = 0
beta_ampa_inter = 0
beta_gaba_pyr = 1
beta_gaba_inh = 1

# Nombre de columnes corticals. Aquest valor canvia a dos si el beta_ampa_inter > 0 en el SIMULATION.py
n_column = 1

# Paràmetres de guany neural
sigma_p = 6.7

# Paràmetres d'adaptació neural
g_k_Na = 1.9

# Espai de paràmetres de la xarxa cortical
c_m = 1.
tau_p, tau_i = 30., 30.
q_max_p, q_max_i = 30e-3, 60e-3
theta_p, theta_i = -58.5, -58.5
sigma_i = 6.
y_e, y_g = 70e-3, 58.6e-3
g_l = 1.
E_p_l, E_i_l = -66., -64.
E_k = -100.
E_ampa, E_gaba = 0., -70.
alpha_Na = 2.
tau_Na = 1.7
R_pump = 0.09
Na_eq = 9.5
Na_pump_0 = R_pump * (Na_eq ** 3 / (Na_eq ** 3 + 3375))
N_pp = 144
N_ii = 40
N_ip = 36
N_pi = 160
N_pINP = 16
N_iINP = 4
N_pP = 16
N_iP = 4
N_Pp = 16
N_Ip = 4
g_ampa = 1
g_gaba = 1
phi_n_sd = 1.2
i_c_m = 1 / c_m

# Definir variables inverses per a càlculs aritmètics més ràpids
i_tau_Na = 1 / tau_Na
i_tau_p, i_tau_i = 1 / tau_p, 1 / tau_i
i_sigma_p = 0.5 * np.pi / sigma_p / np.sqrt(3)
i_sigma_i = 0.5 * np.pi / sigma_i / np.sqrt(3)

# Definir vector de variables per a càlculs vectoritzats més ràpids
INTRA_CONN_EXC = np.array([N_pp, N_ip]).reshape(2, -1)
INTRA_CONN_INH = np.array([N_pi, N_ii]).reshape(2, -1)
INP_CONN_EXC = np.array([N_pINP, N_iINP]).reshape(2, -1)
INTER_CONN_EXC = np.array([N_pP, N_iP]).reshape(2, -1)
beta_gaba = np.array([beta_gaba_pyr, beta_gaba_inh]).reshape(2, -1)

# Definir corrents sinàptics
def I_AMPA(gs, v):
    return g_ampa * gs * (v - E_ampa)

def I_GABA(gs, v):
    return g_gaba * gs * (v - E_gaba)

# Definir taxes de dispar
def Qp(v):
    return 0.5 * q_max_p * (1 + np.tanh((v - theta_p) * i_sigma_p))

def Qi(v):
    return 0.5 * q_max_i * (1 + np.tanh((v - theta_i) * i_sigma_i))

# Definir equacions del camp cortical
def Cortex_Field(yi, input_1):
    # Array buit per a recollir els valors numèrics
    y = np.empty(yi.shape, dtype=float)
    
    q_p, q_i = Qp(yi[0]), Qi(yi[1])
    na_aux = yi[10] * yi[10] * yi[10]
    
    # Calcular corrents sinàptics
    i_ampa = I_AMPA(beta_ampa_intra * yi[[2, 6]] + beta_ampa_input * yi[[11, 13]] + beta_ampa_inter * yi[[15, 17]], yi[:2])
    i_gaba = I_GABA(beta_gaba * yi[[4, 8]], yi[:2])
    
    # Dinàmica en el potencial de membrana de les poblacions excitadores
    y[0] = (-g_l * (yi[0] - E_p_l) - i_ampa[0] - i_gaba[0]) * i_tau_p - g_k_Na * 0.37 / (
                1 + (38.7 / yi[10]) ** 3.5) * (yi[0] - E_k) * i_c_m
    # Dinàmica en el potencial de membrana de les poblacions inhibidores
    y[1] = (-g_l * (yi[1] - E_i_l) - i_ampa[1] - i_gaba[1]) * i_tau_i
    
    # Dinàmica en els corrents sinàptics excitadors i inhibidors degut a les connexions locals
    y[2:9:2] = yi[3:10:2]
    y[[3, 7]] = y_e * (y_e * (INTRA_CONN_EXC * q_p - yi[[2, 6]]) - 2 * yi[[3, 7]])
    y[[5, 9]] = y_g * (y_g * (INTRA_CONN_INH * q_i - yi[[4, 8]]) - 2 * yi[[5, 9]])
    
    # Dinàmica en la concentració de Na per a corrents d'adaptació
    y[10] = (alpha_Na * q_p - (R_pump * (na_aux / (na_aux + 3375)) - Na_pump_0)) * i_tau_Na
    
    # Dinàmica en els corrents sinàptics excitadors degut a la presentació d'estímuls a través de connexions inter excitadores
    y[[11, 13]] = yi[[12, 14]]
    y[[12, 14]] = y_e * (y_e * (INP_CONN_EXC * input_1 - yi[[11, 13]]) - 2 * yi[[12, 14]])
    
    # Dinàmica en els corrents sinàptics excitadors degut a connexions inter excitadores
    y[[15, 17]] = yi[[16, 18]]
    y[[16, 18]] = y_e * (y_e * (INTER_CONN_EXC * q_p[::-1] - yi[[15, 17]]) - 2 * yi[[16, 18]])
    return y

# Definir la integració de segon ordre de Runge-Kutta per al camp cortical
def RK2order_Cor(dt, data_cor, l, input_1):
    k1_cor = dt * Cortex_Field(data_cor, input_1)
    k2_cor = dt * Cortex_Field(data_cor + k1_cor + l, input_1)
    return data_cor + 0.5 * (k1_cor + k2_cor) + l

#Definir els nous valors de beta
def calculate_betas(time):

    max_value = 1
    min_value = 2

    # Calculem el rang de temps on volem aplicar el "step"
    step_start =1  # Segons
    step_end = 9    # Segons

    # Càlcul del temps en ms
    step_start_ms = step_start * 1000 / dt 
    step_end_ms = step_end * 1000 / dt

    # Càlcul de beta_ampa_intra
    growth_rate = (max_value - min_value) / ((step_end - step_start) * 1000 / dt)

    if time < step_start_ms:
        return min_value
    elif step_start_ms <= time <= step_end_ms:
        beta_ampa_intra = min_value + growth_rate * (time - step_start_ms) 
        return beta_ampa_intra
    else:
        return max_value

   
#
#
# Definir la funció per a una integració de prova a través del temps de simulació
def ONE_TRIAL_INTEGRATION(RNG_init,RNG,ex_input,stochastic):
    #
    data_collect = np.zeros((19,n_column,n),dtype=float)
    data_initi = np.empty((19, n_column,2), dtype=float)
    l1 = np.zeros((19,n_column),dtype=float)
    #
    data_initi[0,:,0] = -10*RNG_init[0].random(n_column) + theta_p
    data_initi[1,:,0] = -10*RNG_init[0].random(n_column) + theta_i
    data_initi[2:11,:,0] = 0.01*RNG_init[0].random((9,n_column))
    data_initi[11:19,:,0] = 0
    for i in range(n_initial - 1):
        beta_gaba = np.array([beta_gaba_pyr,beta_gaba_inh]).reshape(2,-1) 
        
        # Runge–Kutta, segon ordre 
        l1[[3,7]] = y_e * y_e * np.sqrt(dt) * np.array([[RNG_init[k+kk*n_column].normal(0,phi_n_sd) for k in range(n_column)]for kk in range(2)]) * stochastic
        data_initi[:,:, 1] = RK2order_Cor(dt,data_initi[:,:, 0],l1,0,beta_ampa_intra,beta_ampa_input,beta_ampa_inter,beta_gaba)
        data_initi[:, :,0] = data_initi[:,:, 1]
    data_collect[:,:,0] = data_initi[:,:,0]
  
    for i in range(n - 1):
        
        beta_ampa_intra= calculate_betas(i)
        beta_gaba = np.array([beta_gaba_pyr,beta_gaba_inh]).reshape(2,-1) 
        if beta_ampa_intra > 1:
            fixedpoint=np.load("../data/V_WAKE.npy")
            beta_gaba= FIND_BETA_GABA(beta_ampa_intra,beta_ampa_inter,fixedpoint)
        
        l1[[3,7]] = y_e * y_e * np.sqrt(dt) * np.array([[RNG[k+kk*n_column].normal(0,phi_n_sd) for k in range(n_column)]for kk in range(2)]) * stochastic
        #
        if i >= n_1 and i < n_1+sens_input_iter: 
            input_1 = ex_input
        else:
            input_1 = 0
        data_collect[:,:, i+1] = RK2order_Cor(dt,data_collect[:,:, i],l1,input_1,beta_ampa_intra,beta_ampa_input,beta_ampa_inter,beta_gaba)
    
    return data_collect

# Definir funció per guarda beta_gaba per contrarestar la sobreexcitació deguda a l'escalat sinàptic
def STORE_beta_gaba():
    beta_gaba_values=[]  
    for i in range(n_initial - 1):
        beta_gaba = np.array([beta_gaba_pyr,beta_gaba_inh]).reshape(2,-1) 
    for i in range(n - 1):
        
        beta_ampa_intra= calculate_betas(i)
        beta_gaba = np.array([beta_gaba_pyr,beta_gaba_inh]).reshape(2,-1) 
        if beta_ampa_intra >=1:
            fixedpoint=np.load("../data/V_WAKE.npy")
            beta_gaba= FIND_BETA_GABA(beta_ampa_intra,beta_ampa_inter,fixedpoint)

        beta_gaba_values.append(beta_gaba[0])
    return beta_gaba_values

# Definir funció per a la simulació d'un assaig al llarg del temps de simulació
def TRIAL_SIMULATION(ex_input,n_trial,stochastic=True):
    RNG = [np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed,spawn_key=(0,0,0,n_column,int(10*phi_n_sd),int(100*ex_input),trial_id,k),)) for k in range(n_trial * 2 * n_column)]
    RNG = [RNG[j*n_column*2:(j+1)*n_column*2] for j in range(n_trial)]
    #
    #
    #
    #
    RNG_init = [np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed,spawn_key=(0,0,1,n_column,int(10*phi_n_sd),int(100*ex_input),trial_id,k),)) for k in range(n_trial * 2 * n_column)]
    RNG_init = [RNG_init[j*n_column*2:(j+1)*n_column*2] for j in range(n_trial)]
    #
    #
    #
    #
    # building variables
    data = np.zeros((n_trial,19,n_column,n),dtype=float)
    # simulation for n trials

    for j in range(n_trial):

        
        time = j * n_initial * dt
        data[j] = ONE_TRIAL_INTEGRATION(RNG_init[j],RNG[j],ex_input,stochastic)

        
     
    return data

# Definir funció per trobar beta_gaba per contrarestar la sobreexcitació deguda a l'escalat sinàptic
def FIND_BETA_GABA(beta_ampa_intra,beta_ampa_inter,fixedpoint): 
    vp = fixedpoint[0]
    vi = fixedpoint[1]
    qp,qi = Qp(vp),Qi(vi)

    A = alpha_Na*qp/R_pump+(Na_eq**3)/(Na_eq**3+3375)
    NA = np.cbrt(A*3375 / (1-A))
    W = 0.37 / (1+(38.7/NA)**3.5)

    beta_gaba_ex_1 = -(g_l*(vp - E_p_l)+ (beta_ampa_intra * N_pp + beta_ampa_inter * N_pP) * g_ampa *qp*(vp-E_ampa) + tau_p/c_m*g_k_Na*W*(vp-E_k)) / (g_gaba * N_pi*qi*(vp-E_gaba) )
    beta_gaba_in_1 = -(g_l*(vi - E_i_l)+ (beta_ampa_intra * N_ip + beta_ampa_inter * N_iP) * g_ampa *qp*(vi-E_ampa)) / ( g_gaba * N_ii*qi*(vi-E_gaba))

    beta_gaba_ex_2 = -(g_l*(vp - E_p_l)+ (beta_ampa_intra * N_pp + beta_ampa_inter * N_Pp) * g_ampa *qp*(vp-E_ampa) + tau_p/c_m*g_k_Na*W*(vp-E_k)) / (g_gaba * N_pi*qi*(vp-E_gaba) )
    beta_gaba_in_2 = -(g_l*(vi - E_i_l)+ (beta_ampa_intra * N_ip + beta_ampa_inter * N_Ip) * g_ampa *qp*(vi-E_ampa)) / ( g_gaba * N_ii*qi*(vi-E_gaba))

    beta_gaba_ex_1 = beta_gaba_ex_1.max()
    beta_gaba_in_1 = beta_gaba_in_1.max()

    beta_gaba_ex_2 = beta_gaba_ex_2.max()
    beta_gaba_in_2 = beta_gaba_in_2.max()

    beta_gaba = np.array([[beta_gaba_ex_1,beta_gaba_ex_2],[beta_gaba_in_1,beta_gaba_in_2]])

    if beta_ampa_inter == 0:

        return beta_gaba[:,0].reshape(2,-1)

    return beta_gaba

# Definir funció per suavitzar
def SMOOTH(y):
    z=np.linspace(-1.5,1.5,7)
    zz=np.exp(-1/1*z**2)

    zz /= zz.sum()
    zzz=np.convolve(y,zz,mode="same")
    return argrelextrema(zzz, np.greater)[0][-1]

# Definir funció per trobar la mitjana de la taxa de dispar
def FIND_Vp_Vi(firing_p,firing_i):
    C = np.pi/2/np.sqrt(3)
    bins_p = np.arange(0,31)
    bins_i = np.arange(0,61)

    q_p_up_mean_index = np.argmax(np.histogram(firing_p[firing_p>10],bins=bins_p)[0])
    q_p_up_mean = bins_p[q_p_up_mean_index]

    v_p_up_mean = np.arctanh(2*q_p_up_mean/1000/q_max_p - 1) * sigma_p/C + theta_p


    q_i_up_mean_index = np.argmax(np.histogram(firing_i[firing_p>10],bins=bins_i)[0])
    q_i_up_mean = bins_i[q_i_up_mean_index]

    v_i_up_mean = np.arctanh(2*q_i_up_mean/1000/q_max_i - 1) * sigma_i/C + theta_i

    return np.array([v_p_up_mean, v_i_up_mean])

