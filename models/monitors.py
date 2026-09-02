

# Plasticity monitors
def plasticity_mon_full(net):
    return {"pfpc_weights": net.pf_to_pc_BCM.weights_per_conn,
        "pfpc_w_cspk" : net.pf_to_pc_BCM.w_cspk,
        "pfpc_w_BCM" : net.pf_to_pc_BCM.w_BCM,
        "pfpc_theta_M" : net.pf_to_pc_BCM.theta_M,
        "pfpc_dw_cspk": net.pf_to_pc_BCM.dw_cspk,
        "pfpc_dw_BCM": net.pf_to_pc_BCM.dw_BCM,
}

def plasticity_mon_min(net):
    return {"pfpc_weights": net.pf_to_pc_BCM.weights_per_conn}

# Stimulus monitors
def stimulus_mon(net):
    return {"stim.isi": net.stim.current_isi,
        "stim.M_io": net.stim.M_io,
        "stim.M_pf": net.stim.M_pf,
    }       
   
def neuron_pot_mon(net):
    return {
        "pf.I_OU": net.pf.I_OU,
        "pf.rho": net.pf.rho,
        "pc.V": net.pc.V,
        "pc.input": net.pc.input,
        "pc.rho": net.pc.rho,
        "cn.V": net.cn.V,
        "cn.I_PC": net.cn.I_PC,
        "io.V_soma": net.io.neurons.V_soma,
        "io.V_axon": net.io.neurons.V_axon,
        "io.V_dend": net.io.neurons.V_dend,
        "io.input": net.io.neurons.input,
        "io.I_OU": net.io.neurons.I_OU,
    }

def  neuron_spike_mon(net):
    return {
        "pc.spike": net.pc.spike,
        "pc.cspk": net.pc.cspk,
        "cn.spike": net.cn.spike,
        "io.spike": net.io.neurons.spike
    }

def stimulus_testing_mon(net):
    return {
        # Monitors for stimuli
        "stim.isi": net.stim.current_isi,
        "stim_pf_start": net.stim.stim_pf_start,
        "stim_io_start": net.stim.stim_io_start,
        "M_pf": net.stim.M_pf,
        "M_io": net.stim.M_io,

        # Monitors for stimulus effect on neurons
        "pf.I_OU": net.pf.I_OU, 
        "pf.rho": net.pf.rho,
        "pf.I_total": net.pf.I_total,

        "pc.rho": net.pc.rho,
        "pc.spike": net.pc.spike,
        "pc.V": net.pc.V,
        "pc.input": net.pc.input,

        "io.I_OU": net.io.neurons.I_OU,
        "io.V_soma": net.io.neurons.V_soma,
        "io.spike": net.io.neurons.spike,
        "io.I_total": net.io.neurons.I_total,

        # Monitors for plasticity 
        "pfpc_weights": net.pf_to_pc_BCM.weights_per_conn,
    }
def training_monitors(net):
    return{}

def testing_monitors(net):
    return{}

monitor_presets = {
    "plasticity_full": plasticity_mon_full,
    "plasticity_min": plasticity_mon_min,
    "neuron_min": neuron_pot_mon,
    "stimulus": stimulus_mon,
    "neuron_spike": neuron_spike_mon,
    "stimulus_testing": stimulus_testing_mon,
}   


