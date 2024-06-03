import sys
sys.path.append("../MODULE/")

import model_parameters_functions20s as MPF
import numpy as np
import pandas as pd
import pickle
import os
import time
import csv
#
#
#
#
MPF.trial_id = 2
#
DIR = "../data/"
#
MPF.beta_ampa_intra = 1 
MPF.beta_ampa_input = 0
MPF.beta_ampa_inter = 0 
#
stochastic = 1
#
MPF.n_column = 1
#

n_trial = 1
if stochastic:
    n_trial = 2
#
#
ex_input = 0 
data = MPF.TRIAL_SIMULATION(ex_input,n_trial,stochastic)
#betas=MPF.STORE_beta_gaba()

#
if MPF.n_column == 2:
    data_to_save = {"pert":
                        {"pyr":1000*MPF.Qp(data[0,0,0,:]),  #células piramidales [0] i Qp, inh[1]i Qi
                        "inh":1000*MPF.Qi(data[0,1,0,:])},
    
                    "unpert":{"pyr":1000*MPF.Qp(data[0,0,1,:]),
                            "inh":1000*MPF.Qi(data[0,1,1,:])}}
   



else:
    data_to_save = {"pert":
                        {"pyr":
                            {"pre":1000*MPF.Qp(data[0,0,0,:])},   #células piramidales [0] i Qp, inh[1]i Qi
                        "inh":{"pre":1000*MPF.Qi(data[0,1,0,:])}}}

   

df = pd.DataFrame(data_to_save)
df.to_pickle(DIR+"data_stochastic_firing_{}_nColumn_{}_Time_{}_beta_ampa_intra_{}.pkl".format(stochastic,MPF.n_column,MPF.beta_ampa_intra)) # Guardar el DataFrame com un archiu pickle

#gb = pd.DataFrame({"beta_gaba":betas})  # Crear un DataFrame con los valores de beta_gaba
#gb.to_pickle(DIR + "beta_gaba_values_{}s_trial_id_{}_bintra_{}_.pkl".format(MPF.T,MPF.trial_id,MPF.beta_ampa_intra))  # Guardar el DataFrame de beta gaba com un archiu pickle
#
#
