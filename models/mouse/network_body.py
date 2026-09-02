import sys

sys.path.append('C:/Users/HP/PycharmProjects/Internproject 2025/cerebellum-jax-main/cerebellum-jax-main')

import matplotlib.pyplot as plt
import itertools
import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.lax as lax
import json
import os

from models import network_dyn as net
from models.mouse.eigenmode_mjx import Body

from utils.connectivity import (
    generate_pf_pc_connectivity,
    generate_pc_cn_connectivity,
    generate_cn_io_connectivity,
    generate_io_pc_connectivity,
)

class AnglesToPC(bp.dyn.SynConn):
    def __init__(self, pre: Body, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
        self.weights = bm.Variable(kwargs['weights'])  # shape: (num_pc, num_pf)
        self.pre_indices_flat = self.conn.require('pre_ids') # this variable lists all indices for muscoskeletal parts
        self.post_indices_flat = self.conn.require('post_ids')  # shape: (num_connections,)
        self.num_connections = len(self.pre_indices_flat)

        self.angle_amp = kwargs.get('angle_amp', 1.),
        self.angle_mu = kwargs.get('angle_mu', 0.),
        if len(self.pre_indices_flat) != len(self.post_indices_flat):
            raise ValueError('PFtoPC connection error: pre_ids and post_ids length mismatch.')

    def update(self):
        pre_I = self.pre.output #pre_I = vector with each element corresponding to muscoskeletal variable, e.g. limb
        pre_I_per_conn = bm.take(pre_I, self.pre_indices_flat) # This vector describes each synapse
        weights_per_conn = self.weights[self.post_indices_flat, self.pre_indices_flat]
        breakpoint()
        contribution_per_conn = ((1 / 5.0) * weights_per_conn * pre_I_per_conn)
        total_input = bm.segment_sum(
            contribution_per_conn,
            self.post_indices_flat,
            num_segments=self.post.num,
        )                                   # total input to each PC is the sum of all activity in each muscoskel segment
        self.post.input.value = total_input  # shape: (num_pc,)

class Mouse(bp.DynSysGroup):
    def __init__(self, num_pf_bundles=5, num_pc=100, num_cn=40, num_io=64, **kwargs):
        super().__init__()

        # --- Central Parameter Definition --- #

        # Population sizes
        self.num_pf_bundles = num_pf_bundles
        self.num_pc = num_pc
        self.num_cn = num_cn
        self.num_io = num_io


        # PF parameters
        pf_params = {
            "PF_I_OU0": kwargs.get("PF_I_OU0", 1.3 ), # nA
            "PF_tau_OU": kwargs.get("PF_tau_OU", 50.0),
            "PF_sigma_OU": kwargs.get("PF_sigma_OU", 0.25),
        }



        # PC parameters
        pc_params = {
            "C": bm.random.normal(
                kwargs.get("PC_C_mean", 75.0), kwargs.get("PC_C_std", 1.0), num_pc
            ),
            "gL": bm.random.normal(
                kwargs.get("PC_gL_mean", 30.0), kwargs.get("PC_gL_std", 1.0), num_pc
            )
            * 0.001,  # nS to microS
            "EL": bm.random.normal(
                kwargs.get("PC_EL_mean", -70.6), kwargs.get("PC_EL_std", 0.5), num_pc
            ),
            "VT": bm.random.normal(
                kwargs.get("PC_VT_mean", -50.4), kwargs.get("PC_VT_std", 0.5), num_pc
            ),
            "DeltaT": bm.random.normal(
                kwargs.get("PC_DeltaT_mean", 2.0),
                kwargs.get("PC_DeltaT_std", 0.5),
                num_pc,
            ),
            "tauw": bm.random.normal(
                kwargs.get("PC_tauw_mean", 144.0),
                kwargs.get("PC_tauw_std", 2.0),
                num_pc,
            ),
            "a": bm.random.normal(
                kwargs.get("PC_a_mean", 4.0), kwargs.get("PC_a_std", 0.5), num_pc
            )
            * 0.001,  # nS to microS
            "b": bm.random.normal(
                kwargs.get("PC_b_mean", 0.0805), kwargs.get("PC_b_std", 0.001), num_pc
            ),
            "Vr": bm.random.normal(
                kwargs.get("PC_Vr_mean", -70.6), kwargs.get("PC_Vr_std", 0.5), num_pc
            ),
            "I_intrinsic": bm.random.normal(
                kwargs.get("PC_I_intrinsic_mean", 0.35),
                kwargs.get("PC_I_intrinsic_std", 0.21),
                num_pc,
            ),
            "v_init": bm.random.normal(
                kwargs.get("PC_v_init_mean", -70.6),
                kwargs.get("PC_v_init_std", 0.5),
                num_pc,
            ),
            "w_init": bm.zeros(num_pc)
            * kwargs.get("PC_w_init_val", 0.0),  # Allow setting via kwarg if needed
            "tau_rate": kwargs.get("PC_tau_rate", 200.0), # ms
        }

        # CN parameters
        cn_params = {
            "C": bm.random.normal(
                kwargs.get("CN_C_mean", 281.0), kwargs.get("CN_C_std", 1.0), num_cn
            ),
            "gL": bm.random.normal(
                kwargs.get("CN_gL_mean", 30.0), kwargs.get("CN_gL_std", 1.0), num_cn
            )
            * 0.001,  # nS to microS
            "EL": bm.random.normal(
                kwargs.get("CN_EL_mean", -70.6), kwargs.get("CN_EL_std", 0.5), num_cn
            ),
            "VT": bm.random.normal(
                kwargs.get("CN_VT_mean", -50.4), kwargs.get("CN_VT_std", 0.5), num_cn
            ),
            "DeltaT": bm.random.normal(
                kwargs.get("CN_DeltaT_mean", 2.0),
                kwargs.get("CN_DeltaT_std", 0.5),
                num_cn,
            ),
            "tauw": bm.random.normal(
                kwargs.get("CN_tauw_mean", 30.0), kwargs.get("CN_tauw_std", 1.0), num_cn
            ),
            "a": bm.random.normal(
                kwargs.get("CN_a_mean", 4.0), kwargs.get("CN_a_std", 0.5), num_cn
            )
            * 0.001,  # nS to microS
            "b": bm.random.normal(
                kwargs.get("CN_b_mean", 0.0805), kwargs.get("CN_b_std", 0.001), num_cn
            ),
            "Vr": bm.random.normal(
                kwargs.get("CN_Vr_mean", -65.0), kwargs.get("CN_Vr_std", 0.5), num_cn
            ),
            "I_intrinsic": bm.ones(num_cn) * kwargs.get("CN_I_intrinsic_val", 1.2),
            "v_init": bm.random.normal(
                kwargs.get("CN_v_init_mean", -65.0),
                kwargs.get("CN_v_init_std", 3.0),
                num_cn,
            ),
            "w_init": bm.zeros(num_cn) * kwargs.get("CN_w_init_val", 0.0),
            "tauI": bm.random.normal(
                kwargs.get("CN_tauI_mean", 30.0), kwargs.get("CN_tauI_std", 1.0), num_cn
            ),
        }

        # IO Neuron parameters (passed to IONetwork)
        io_neuron_params = {
            "g_Na_s": bm.random.normal(
                kwargs.get("IO_g_Na_s_mean", 150.0),
                kwargs.get("IO_g_Na_s_std", 1.0),
                num_io,
            ),  # mS/cm2
            "g_CaL": kwargs.get("IO_g_CaL_base", 0.5)
            + kwargs.get("IO_g_CaL_factor", 1.2) * bm.random.rand(num_io),  # mS/cm2
            "g_Kdr_s": bm.random.normal(
                kwargs.get("IO_g_Kdr_s_mean", 9.0),
                kwargs.get("IO_g_Kdr_s_std", 0.1),
                num_io,
            ),  # mS/cm2
            "g_K_s": bm.random.normal(
                kwargs.get("IO_g_K_s_mean", 5.0),
                kwargs.get("IO_g_K_s_std", 0.1),
                num_io,
            ),  # mS/cm2
            "g_h": bm.random.normal(
                kwargs.get("IO_g_h_mean", 0.12), kwargs.get("IO_g_h_std", 0.01), num_io
            ),
            "g_ls": bm.random.normal(
                kwargs.get("IO_g_ls_mean", 0.017),
                kwargs.get("IO_g_ls_std", 0.001),
                num_io,
            ),  # mS/cm2
            "g_CaH": bm.random.normal(
                kwargs.get("IO_g_CaH_mean", 4.5),
                kwargs.get("IO_g_CaH_std", 0.1),
                num_io,
            ),  # mS/cm2
            "g_K_Ca": bm.random.normal(
                kwargs.get("IO_g_K_Ca_mean", 35.0),
                kwargs.get("IO_g_K_Ca_std", 0.5),
                num_io,
            ),  # mS/cm2
            "g_ld": bm.random.normal(
                kwargs.get("IO_g_ld_mean", 0.016),
                kwargs.get("IO_g_ld_std", 0.001),
                num_io,
            ),  # mS/cm2
            "g_Na_a": bm.random.normal(
                kwargs.get("IO_g_Na_a_mean", 240.0),
                kwargs.get("IO_g_Na_a_std", 1.0),
                num_io,
            ),  # mS/cm2
            "g_K_a": bm.random.normal(
                kwargs.get("IO_g_K_a_mean", 240.0),
                kwargs.get("IO_g_K_a_std", 0.5),
                num_io,
            ),  # mS/cm2
            "g_la": bm.random.normal(
                kwargs.get("IO_g_la_mean", 0.017),
                kwargs.get("IO_g_la_std", 0.001),
                num_io,
            ),  # mS/cm2
            "V_Na": bm.random.normal(
                kwargs.get("IO_V_Na_mean", 55.0), kwargs.get("IO_V_Na_std", 1.0), num_io
            ),  # mV
            "V_Ca": bm.random.normal(
                kwargs.get("IO_V_Ca_mean", 120.0),
                kwargs.get("IO_V_Ca_std", 1.0),
                num_io,
            ),  # mV
            "V_K": bm.random.normal(
                kwargs.get("IO_V_K_mean", -75.0), kwargs.get("IO_V_K_std", 1.0), num_io
            ),  # mV
            "V_h": bm.random.normal(
                kwargs.get("IO_V_h_mean", -43.0), kwargs.get("IO_V_h_std", 1.0), num_io
            ),  # mV
            "V_l": bm.random.normal(
                kwargs.get("IO_V_l_mean", 10.0), kwargs.get("IO_V_l_std", 1.0), num_io
            ),  # mV
            "S": bm.random.normal(
                kwargs.get("IO_S_mean", 1.0), kwargs.get("IO_S_std", 0.1), num_io
            ),  # 1/C_m, cm^2/uF
            "g_int": bm.random.normal(
                kwargs.get("IO_g_int_mean", 0.13),
                kwargs.get("IO_g_int_std", 0.001),
                num_io,
            ),  # Cell internal conductance - no unit given
            "p1": bm.random.normal(
                kwargs.get("IO_p1_mean", 0.25), kwargs.get("IO_p1_std", 0.01), num_io
            ),  # Cell surface ratio soma/dendrite - no unit given
            "p2": bm.random.normal(
                kwargs.get("IO_p2_mean", 0.15), kwargs.get("IO_p2_std", 0.01), num_io
            ),  # Cell surface ratio axon(hillock)/soma - no unit given
            "I_OU0": bm.asarray(kwargs.get("IO_I_OU0", -0.3)),  # mA/cm2
            "tau_OU": bm.asarray(kwargs.get("IO_tau_OU", 50.0)),  # ms
            "sigma_OU": bm.asarray(kwargs.get("IO_sigma_OU", 0.3)),  # mV

            "io_threshold": kwargs.get("IO_threshold", -30.0), # mV

            # Initial states
            "V_soma_init": bm.random.normal(
                kwargs.get("IO_V_soma_init_mean", -60.0),
                kwargs.get("IO_V_soma_init_std", 3.0),
                num_io,
            ),  # mV
            "V_axon_init": bm.random.normal(
                kwargs.get("IO_V_axon_init_mean", -60.0),
                kwargs.get("IO_V_axon_init_std", 3.0),
                num_io,
            ),  # mV
            "V_dend_init": bm.random.normal(
                kwargs.get("IO_V_dend_init_mean", -60.0),
                kwargs.get("IO_V_dend_init_std", 3.0),
                num_io,
            ),  # mV
            # Apparentely, all these initial values need to be exactly the same for all IO neurons
            # Otherwise, IOs explode
            "soma_k_init": 0.7423159
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_l_init": 0.0321349
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_h_init": 0.3596066
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_n_init": 0.2369847
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_x_init": 0.1
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "axon_Sodium_h_init": 0.9
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "axon_Potassium_x_init": 0.2369847
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Ca2Plus_init": 3.715
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Calcium_r_init": 0.0113
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Potassium_s_init": 0.0049291
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Hcurrent_q_init": 0.0337836
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
        }

        # IO Network parameters
        ionet_params = {
            "g_gj": kwargs.get("IO_g_gj", 0.05),
            "n_clusters": kwargs.get("IO_n_clusters", 1),
            "n_projections": kwargs.get("IO_n_projections", 4),
            "p_bridge": kwargs.get("IO_bridge_probability", 0),
        }

        # Stimulus parameters
        stim_params = {
            "OU_stim_isi_mean": kwargs.get("OU_stim_isi_mean", 120.0), # ms
            "OU_stim_isi_std": kwargs.get("OU_stim_isi_std", 0.0), # ms
            "OU_stim_freq": kwargs.get("OU_stim_freq", 700.0), 
            "OU_stim_start": kwargs.get("OU_stim_start", 200.0),

            "OU_stim_dur_io_mean": kwargs.get("OU_stim_dur_io_mean", 500.0),
            "OU_stim_dur_pf_mean": kwargs.get("OU_stim_dur_pf_mean", 500.0),
            "OU_stim_dur_io_std": kwargs.get("OU_stim_dur_io_std", 0.0),
            "OU_stim_dur_pf_std":kwargs.get( "OU_stim_dur_pf_std", 0.0),

            "OU_stim_amp_io_mean":kwargs.get( "OU_stim_amp_io_mean", 1.4),
            "OU_stim_amp_pf_mean": kwargs.get( "OU_stim_amp_pf_mean", 1.4 ), # 5 % increase from baseline
            "OU_stim_amp_io_std": kwargs.get("OU_stim_amp_io_std" , 0.0),
            "OU_stim_amp_pf_std": kwargs.get("OU_stim_amp_pf_std" , 0.0),

            "OU_stim_pf_on": kwargs.get("OU_stim_pf_on", False),
            "OU_stim_io_on": kwargs.get("OU_stim_io_on", False),
        }


        # Synapse parameters
        pfpc_params = {
            "theta_M_init": kwargs.get("PFPC_theta_M_init", 0.060 ), # kHz 0.060
            "A_cspk": kwargs.get("PFPC_A_cspk", -0.1),
            "tau_M": kwargs.get("PFPC_tau_M", 15.0), # ms 15.0
            "tau_cspk": kwargs.get("PFPC_tau_cspk", 350.0),
            "I_PF_0" : kwargs.get("PFPC_IO_I_PF_0", 0.0013), # kHz
            "pf_scaling": kwargs.get("PFPC_pf_scaling", 0.0050),
            "plasticity_on": kwargs.get("PFPC_plasticity_on", True),
        }

        pccn_params = {
            "delay": kwargs.get("PCCN_delay", 10.0),
            "gamma_PC": kwargs.get("PCCN_gamma_PC", 0.004),
        }
        cnio_params = {
            "delay": kwargs.get("CNIO_delay", 50.0),
            "tau_inhib": kwargs.get("CNIO_tau_inhib", 30.0),
            "gamma_CN_IO": kwargs.get("CNIO_gamma_CN_IO", -1.8),
        }
        iopc_params = {
            "delay": kwargs.get("IOPC_delay", 15.0),
            "cs_weight": kwargs.get("IOPC_cs_weight", 0.22),
            "io_threshold": kwargs.get("IOPC_io_threshold", -30.0),
        }


        # --- Create Populations --- #
        self.pf = net.PFBundles(num_bundles=num_pf_bundles, **pf_params)
        self.pc = net.PurkinjeCell(num_pc, **pc_params)
        self.cn = net.DeepCerebellarNuclei(num_cn, **cn_params)
        io_params = {**ionet_params, **io_neuron_params}
        self.io = net.IONetwork(num_neurons=num_io, **io_params)

        self.body = Body(delta=kwargs.get('delta', None))

        # --- Set up Connectivity --- #
        pfpc_pre, pfpc_post, pfpc_weights = generate_pf_pc_connectivity(num_pf_bundles, num_pc)
        pfpc_conn = bp.conn.IJConn(pfpc_pre, pfpc_post)
    
        pfpc_params["init_weights"] = pfpc_weights[pfpc_post, pfpc_pre] # Add generated weights
        pfpc_params["n_connections"] = len(pfpc_pre)
        pfpc_params["pre_pf_idx"]= pfpc_pre
        pfpc_params["post_pc_idx"] = pfpc_post

        pccn_pre, pccn_post = generate_pc_cn_connectivity(num_pc, num_cn)
        pccn_conn = bp.conn.IJConn(pccn_pre, pccn_post)

        cnio_pre, cnio_post = generate_cn_io_connectivity(num_cn, num_io)
        cnio_conn = bp.conn.IJConn(cnio_pre, cnio_post)

        iopc_pre, iopc_post = generate_io_pc_connectivity(num_io, num_pc)
        iopc_conn = bp.conn.IJConn(iopc_pre, iopc_post)

        self.pre_io_idx = iopc_pre
        self.post_pc_idx = iopc_post

        # --- Create Synapses --- #
        self.io_to_pc = net.IOToPC(                 # IOtoPC first, to use post.cpsk value in PFtoPC
                    pre=self.io.neurons, post=self.pc, conn=iopc_conn, **iopc_params
                )

        self.body_to_pc = AnglesToPC(
                    pre=self.body, post=self.pc, conn= pfpc_conn, weights=pfpc_weights,
                    angle_amp=kwargs.get('angle_amp', 1.),
                    angle_mu=kwargs.get('angle_mu', 0.),
                    )
        if kwargs.get("pf_to_pc", True):
            self.pf_to_pc_BCM = net.PFtoPC_BCM(size= pfpc_params["n_connections"], pre=self.pf, post=self.pc, pre_idx=pfpc_pre , post_idx=pfpc_post, **pfpc_params)
            self.pf_to_pc = net.PFtoPC(pre=self.pf, post=self.pc, conn=pfpc_conn, syndyn=self.pf_to_pc_BCM, **pfpc_params)

        self.pc_to_cn = net.PCToCN(pre=self.pc, post=self.cn, conn=pccn_conn, **pccn_params)
        self.cn_to_io = net.CNToIO(
            pre=self.cn, post=self.io.neurons, conn=cnio_conn, **cnio_params
        )
