#!/usr/bin/env python3

import pickle
import numpy as np

data_rh = np.load("data/data_df_c_RH.npy", allow_pickle=True).item()
data_av = np.load("data/data_df_c_AV.npy", allow_pickle=True).item()

with open('data/data_df_full_RH.pickle', 'rb') as f:
    data_rhdf = pickle.load(f)
    
with open('data/raw_dfs_AV.pkl', 'rb') as f:
    df_av = pickle.load(f)
    
with open('data/raw_dfs_RH.pkl', 'rb') as f:
    df_rh = pickle.load(f)