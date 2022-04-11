import djlib as dj
import numpy as np
import json
import pickle

with open("{settings_file}") as f:
    settings = json.load(f)

grid_size = settings["grid_size"]
mu_vec = settings["mu_vec"]
t_vec = settings["t_vec"]
v_nn = settings["v_nn"]
v_point = settings["v_point"]


data = dj.gc_mc.run_mc_temp_mu_sweep(mu_vec, t_vec, v_point, v_nn, grid_size)
with open("mc_results.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
