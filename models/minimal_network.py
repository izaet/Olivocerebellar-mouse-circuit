import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
x

import warnings
import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.lax as lax

from models.cells.io_clust import IONetwork
from utils.connectivity import generate_cn_io_connectivity



class DeepCerebellarNuclei(bp.dyn.NeuDyn):
    def __init__(self, size, **kwargs):
        super().__init__(size=size)

        self.spike_start = kwargs['spike_start']
        self.mean_period= kwargs[ 'mean_period']
        self.period_std = kwargs[ 'period_std']
        self.spike_duration = kwargs['spike_duration']

        self.spike= bm.Variable(bm.zeros(self.num, dtype=bool))
        self.in_spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.spike_end = bm.Variable(bm.zeros(self.num) * -1)

        if self.spike_start is None:
            self.next_spike = bm.Variable(bm.ones(self.num) * bm.inf)
        else:
            self.next_spike = bm.Variable(bm.ones(self.num) * self.spike_start)

    def update(self):
        t = bp.share['t']
        spike_starting = (t >= self.next_spike) & (~self.in_spike)

        self.in_spike.value = self.in_spike | spike_starting
        self.spike_end.value = bm.where(spike_starting, t + self.spike_duration, self.spike_end.value) # If spike starting, update time when spike ends

        spike_ending = self.in_spike & (t >= self.spike_end) # False when time exceeds spike end
        self.in_spike.value = self.in_spike & (~spike_ending)

        isi = self.mean_period + self.period_std * bm.random.randn(self.num)
        self.next_spike.value = bm.where(spike_ending, t+isi, self.next_spike.value) # If spike ends, update next spike time

        self.spike.value = self.in_spike




class CNToIO(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.tau_inhib = kwargs["tau_inhib"]
        self.gamma_CN_IO = kwargs["gamma_CN_IO"]
        self.delay = kwargs["delay"]
        # indices, indptr for pre->post mapping
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        # self.post_indices shape: (num_connections,)
        # self.post_indptr shape: (num_pre + 1,)
        self.delay_length = int(self.delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)
        self.I_cn = bm.Variable(bm.zeros(post.num))

        # Calculate N_CN (number of CN inputs) for each IO cell
        (_, post_indptr_for_norm) = self.conn.require("post2pre")
        n_cn_per_io_np = np.diff(np.asarray(post_indptr_for_norm))

        if len(n_cn_per_io_np) < post.num:
            temp_n_cn = np.zeros(post.num, dtype=int)
            temp_n_cn[: len(n_cn_per_io_np)] = n_cn_per_io_np
            self.n_cn_per_io = bm.asarray(temp_n_cn)
        else:
            self.n_cn_per_io = bm.asarray(
                n_cn_per_io_np[: post.num]
            )  # Ensure it doesn't exceed post.num

        # Precompute mapping from connection index to source presynaptic index
        self.num_connections = len(self.post_indices)
        source_indices_per_conn_np = np.zeros(self.num_connections, dtype=np.uint32)
        post_indptr_np = np.asarray(self.post_indptr)
        for i in range(self.pre.num):
            start, end = post_indptr_np[i], post_indptr_np[i + 1]
            source_indices_per_conn_np[start:end] = i
        self.source_indices_per_conn = bm.asarray(
            source_indices_per_conn_np
        )  # shape: (num_connections,)

        # Precompute N_CN for the target IO of each connection
        post_indices_np = np.asarray(self.post_indices)
        # Clamp N_CN to minimum 1 to avoid division by zero
        self.target_n_cn_per_conn = bm.maximum(
            self.n_cn_per_io[post_indices_np], 1.0
        ).astype(
            bm.float32
        )  # shape: (num_connections,)

    def update(self):
        dt = bp.share["dt"]

        # 1. Apply exponential decay based on Eq. (23)
        decay_factor = bm.exp(-dt / self.tau_inhib)
        self.I_cn.value *= decay_factor

        # 2. Process delayed spikes and calculate increments based on Eq. (24)
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_pre,) Boolean

        # Check which connections originated from a spiking neuron
        source_spiked_mask = bm.take(
            delayed_spikes, self.source_indices_per_conn
        )  # shape: (num_connections,) Boolean

        # Calculate the increment PER SPIKING CONNECTION (will be negative)
        potential_increment = (
            self.gamma_CN_IO / self.target_n_cn_per_conn
        )  # gamma is negative
        connection_increments = bm.where(
            source_spiked_mask, potential_increment, 0.0
        )  # shape: (num_connections,)

        # Sum increments for each target postsynaptic neuron (IO cell)
        I_cn_increase = bm.segment_sum(
            connection_increments,
            self.post_indices,  # Target IO indices as Segment IDs
            num_segments=self.post.num,  # Output shape: (num_io,)
        )

        # 3. Add the increments to the current state
        self.I_cn.value += I_cn_increase

        # 4. Assign the total inhibitory current to the postsynaptic input variable
        #    This OVERWRITES any previous value in post.input from this synapse
        self.post.input.value = self.I_cn.value

class IOinhibitionNetwork(bp.DynSysGroup):
    def __init__(self, num_cn=1, num_io=10, **kwargs):
        super(IOinhibitionNetwork, self).__init__()

        # --- Central Parameter Definition --- #

        # Population sizes
        self.num_cn = num_cn
        self.num_io = num_io

        # Dummy CN Neuron parameters
        cn_neuron_params = {
            "mean_period": kwargs.get ("CN_isi_mean", 500.0),
            "period_std": kwargs.get ("CN_isi_std", 50.0),
            "spike_start": kwargs.get ("CN_start_spikes", 1000.0),
            "spike_duration": kwargs.get("CN_spike_duration", 2.0),
        }


        # IO Neuron parameters (passed to IONetwork)
        io_neuron_params = {
            "g_Na_s": bm.random.normal(
                kwargs.get("IO_g_Na_s_mean", 150.0),
                kwargs.get("IO_g_Na_s_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_CaL": kwargs.get("IO_g_CaL_base", 0.5)
                     + kwargs.get("IO_g_CaL_factor", 0.0) * bm.random.rand(num_io),  # mS/cm2
            "g_Kdr_s": bm.random.normal(
                kwargs.get("IO_g_Kdr_s_mean", 9.0),
                kwargs.get("IO_g_Kdr_s_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_K_s": bm.random.normal(
                kwargs.get("IO_g_K_s_mean", 5.0),
                kwargs.get("IO_g_K_s_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_h": bm.random.normal(
                kwargs.get("IO_g_h_mean", 0.12), kwargs.get("IO_g_h_std", 0.00), num_io
            ),
            "g_ls": bm.random.normal(
                kwargs.get("IO_g_ls_mean", 0.017),
                kwargs.get("IO_g_ls_std", 0.000),
                num_io,
            ),  # mS/cm2
            "g_CaH": bm.random.normal(
                kwargs.get("IO_g_CaH_mean", 4.5),
                kwargs.get("IO_g_CaH_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_K_Ca": bm.random.normal(
                kwargs.get("IO_g_K_Ca_mean", 35.0),
                kwargs.get("IO_g_K_Ca_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_ld": bm.random.normal(
                kwargs.get("IO_g_ld_mean", 0.016),
                kwargs.get("IO_g_ld_std", 0.000),
                num_io,
            ),  # mS/cm2
            "g_Na_a": bm.random.normal(
                kwargs.get("IO_g_Na_a_mean", 240.0),
                kwargs.get("IO_g_Na_a_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_K_a": bm.random.normal(
                kwargs.get("IO_g_K_a_mean", 240.0),
                kwargs.get("IO_g_K_a_std", 0.0),
                num_io,
            ),  # mS/cm2
            "g_la": bm.random.normal(
                kwargs.get("IO_g_la_mean", 0.017),
                kwargs.get("IO_g_la_std", 0.000),
                num_io,
            ),  # mS/cm2
            "V_Na": bm.random.normal(
                kwargs.get("IO_V_Na_mean", 55.0), kwargs.get("IO_V_Na_std", 0.0), num_io
            ),  # mV
            "V_Ca": bm.random.normal(
                kwargs.get("IO_V_Ca_mean", 120.0),
                kwargs.get("IO_V_Ca_std", 0.0),
                num_io,
            ),  # mV
            "V_K": bm.random.normal(
                kwargs.get("IO_V_K_mean", -75.0), kwargs.get("IO_V_K_std", 0.0), num_io
            ),  # mV
            "V_h": bm.random.normal(
                kwargs.get("IO_V_h_mean", -43.0), kwargs.get("IO_V_h_std", 0.0), num_io
            ),  # mV
            "V_l": bm.random.normal(
                kwargs.get("IO_V_l_mean", 10.0), kwargs.get("IO_V_l_std", 0.0), num_io
            ),  # mV
            "S": bm.random.normal(
                kwargs.get("IO_S_mean", 1.0), kwargs.get("IO_S_std", 0.0), num_io
            ),  # 1/C_m, cm^2/uF
            "g_int": bm.random.normal(
                kwargs.get("IO_g_int_mean", 0.13),
                kwargs.get("IO_g_int_std", 0.000),
                num_io,
            ),  # Cell internal conductance - no unit given
            "p1": bm.random.normal(
                kwargs.get("IO_p1_mean", 0.25), kwargs.get("IO_p1_std", 0.00), num_io
            ),  # Cell surface ratio soma/dendrite - no unit given
            "p2": bm.random.normal(
                kwargs.get("IO_p2_mean", 0.15), kwargs.get("IO_p2_std", 0.00), num_io
            ),  # Cell surface ratio axon(hillock)/soma - no unit given
            "I_OU0": bm.asarray(kwargs.get("IO_I_OU0", -0.03)),  # mA/cm2
            "tau_OU": bm.asarray(kwargs.get("IO_tau_OU", 50.0)),  # ms
            "sigma_OU": bm.asarray(kwargs.get("IO_sigma_OU", 0.0)),  # mV

            "io_threshold": kwargs.get("IO_threshold", -30.0),  # mV

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

        # Synapse parameters
        cnio_params = {
            "delay": kwargs.get("CNIO_delay", 0),
            "tau_inhib": kwargs.get("CNIO_tau_inhib", 30.0),
            "gamma_CN_IO": kwargs.get("CNIO_gamma_CN_IO", -0.02),
        }

        # --- Create Populations --- #
        io_params = {**ionet_params, **io_neuron_params}
        self.io = IONetwork(num_neurons=num_io, **io_params)
        self.cn = DeepCerebellarNuclei(num_cn, **cn_neuron_params)

        # --- Create Connectivity --- #
        cnio_pre, cnio_post = generate_cn_io_connectivity(num_cn, num_io)
        cnio_conn = bp.conn.IJConn(cnio_pre, cnio_post)
        self.cn_to_io = CNToIO(
            pre=self.cn, post=self.io.neurons, conn=cnio_conn, **cnio_params
        )


def run_simulation(duration=1000.0, dt=0.025, net_params=None, seed=42, jit=True):
    np.random.seed(seed)
    bm.random.seed(seed)

    # Create network instance, passing parameters if provided
    if net_params is None:
        net_params = {}
    # Silence warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    net = IOinhibitionNetwork(**net_params)

    # --- Monitors Configuration --- #
    monitors = {
        # Neuron monitors
        "io.V_soma": net.io.neurons.V_soma,
        "io.V_axon": net.io.neurons.V_axon,
        "io.V_dend": net.io.neurons.V_dend,
        "io.input": net.io.neurons.input,
        "io.I_OU": net.io.neurons.I_OU,
        "io.spike": net.io.neurons.spike,
        "cn.spike": net.cn.spike,
    }

    runner = bp.DSRunner(net, monitors=monitors, dt=dt, jit=jit, progress_bar=True)
    runner.progress_bar = False
    if jit:
        runner._fun_predict = bm.jit(runner._fun_predict)
    runner.run(duration)

    return runner



