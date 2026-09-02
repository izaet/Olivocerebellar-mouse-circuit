import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


import warnings
import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax
import time
import os
import json
import copy

from models.network_dyn import CerebellarNetwork
from models.monitors import monitor_presets


def init_net_and_runner(net_params=None, dt=0.025 , seed=88, jit=True):
    np.random.seed(seed)
    bm.random.seed(seed)

     # Silence warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Create network instance, passing parameters if provided
    if net_params is None:
        net_params = {}

    net = CerebellarNetwork(**net_params)


    # --- Monitors Configuration --- #
    monitor_function = monitor_presets[net_params["monitor_preset"]]
    monitors = monitor_function(net)
    
    runner = bp.DSRunner(net, monitors=monitors, dt=dt, jit=jit, progress_bar=False)
    if jit:
        runner._fun_predict = bm.jit(runner._fun_predict)
    return net, runner
    

def run_simulation(net, runner, duration, downsample= 30):
    runner.run(duration)
    data = {k: np.array(runner.mon[k][::downsample]) for k in runner.mon}
    return net, runner, data
    



    
def run_until_convergence(net, runner, downsample= 30, max_runtime = 500_000, epoch =  250, conv_thresh_m= 0.1, conv_thresh_var = 0.2, chunk_thresh = 1):
    """
    Run the network until PF-PC synapse weights converge. Convergence is defined as the stabilization of the mean andvariance of all synapse weights.
    """
    net.pf_to_pc_BCM.plasticity_on.value= bm.asarray(True)


    runtime= 0.0
    mean_w_previous = np.mean(net.pf_to_pc_BCM.weights_per_conn.value )
    var_w_previous = np.var(net.pf_to_pc_BCM.weights_per_conn.value)
    stable_count = 0
  
    mon_hist = {}

    while runtime < max_runtime:
        
        runner.run(epoch)
        runtime += epoch

        # Append runners for each simulation epoch
        for k  in runner.mon:
            if k not in mon_hist:
                mon_hist[k] = [np.array(runner.mon[k][::downsample])]
            else:
                mon_hist[k].append(np.array(runner.mon[k][::downsample])) 

        # Check for convergence via mean and variance of synapse weights
        w_current = net.pf_to_pc_BCM.weights_per_conn.value
        mean_w_current = np.mean(w_current)
        var_w_current = np.var(w_current)

        d_mean= np.abs(mean_w_current - mean_w_previous)
        d_var = np.abs(var_w_current - var_w_previous)

        if d_mean < conv_thresh_m and d_var < conv_thresh_var:
            stable_count+= 1
        else:
            stable_count = 0
        
        if stable_count >= chunk_thresh:
            break
        
        mean_w_previous = mean_w_current
        var_w_previous = var_w_current

    # Combine all chunks into one runner
    full_mon = {k: np.concatenate (v, axis = 0) for k, v in mon_hist.items()}
            
        
    return net, runner, full_mon, mean_w_current, var_w_current, runtime

    
def init_and_run(duration=1000.0, dt=0.025, net_params=None, seed=42, jit=True):
    np.random.seed(seed)
    bm.random.seed(seed)

    # Create network instance, passing parameters if provided
    if net_params is None:
        net_params = {}
    # Silence warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    net = CerebellarNetwork(**net_params, name="CerebellarNetwork9") 

    # --- Params to return ------- #
    connections_idx = {"pf_pc_pre": net.pf_to_pc_BCM.pre_idx,
                   "pf_pc_post": net.pf_to_pc_BCM.post_idx,
                   "io_pc_pre": net.io_to_pc.io_source_indices,
                   "io_pc_post": net.io_to_pc.pc_target_indptr}

    io_topography_params = {"n_bridges": net.io.n_bridges,
                            "io_src": np.array(net.io.neurons.gj_src),
                            "io_tgt": np.array(net.io.neurons.gj_tgt),
                            "io_cluster_ids": net.io.cluster_ids,
                            "n_neurons": net.num_io}

    # --- Monitors Configuration --- #

    monitor_function = monitor_presets[net_params["monitor_preset"]]
    monitors = monitor_function(net)

    runner = bp.DSRunner(net, monitors=monitors, dt=dt, jit =jit, progress_bar=True)
    runner.progress_bar = False
    if jit:
        runner._fun_predict = bm.jit(runner._fun_predict)
    runner.run(duration)



    return runner, io_topography_params, connections_idx

