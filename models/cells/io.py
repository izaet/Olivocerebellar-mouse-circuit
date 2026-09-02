import jax
import numpy as np
import brainpy as bp
import brainpy.math as bm


class IONeuron(bp.dyn.NeuDyn):
    """
    Inferior Olive (IO) neuron model with three compartments: soma, axon, and dendrite.

    The model includes various ion channels and gap junction connectivity in the dendritic compartment.

    Parameters
    ----------
    size : int
        Number of neurons
    kwargs : dict
        Dictionary containing physiological parameters and initial state values:
        g_int, p1, p2, g_CaL, g_h, g_K_Ca, g_ld, g_la, g_ls, g_Na_s, g_Kdr_s,
        g_K_s, g_CaH, g_Na_a, g_K_a, S, V_Na, V_K, V_Ca, V_h, V_l,
        I_OU0, tau_OU, sigma_OU,
        V_soma_init, soma_k_init, soma_l_init, soma_h_init, soma_n_init, soma_x_init,
        V_axon_init, axon_Sodium_h_init, axon_Potassium_x_init,
        V_dend_init, dend_Ca2Plus_init, dend_Calcium_r_init, dend_Potassium_s_init, dend_Hcurrent_q_init
    """

    def __init__(self, size=1, method="rk4", **kwargs):
        super(IONeuron, self).__init__(size=size)

        # Store parameters
        self.g_int = bm.asarray(kwargs["g_int"])
        self.p1 = bm.asarray(kwargs["p1"])
        self.p2 = bm.asarray(kwargs["p2"])
        self.g_CaL = bm.asarray(kwargs["g_CaL"])
        self.g_h = bm.asarray(kwargs["g_h"])
        self.g_K_Ca = bm.asarray(kwargs["g_K_Ca"])
        self.g_ld = bm.asarray(kwargs["g_ld"])
        self.g_la = bm.asarray(kwargs["g_la"])
        self.g_ls = bm.asarray(kwargs["g_ls"])
        self.g_Na_s = bm.asarray(kwargs["g_Na_s"])
        self.g_Kdr_s = bm.asarray(kwargs["g_Kdr_s"])
        self.g_K_s = bm.asarray(kwargs["g_K_s"])
        self.g_CaH = bm.asarray(kwargs["g_CaH"])
        self.g_Na_a = bm.asarray(kwargs["g_Na_a"])
        self.g_K_a = bm.asarray(kwargs["g_K_a"])
        self.S = bm.asarray(kwargs["S"])
        self.V_Na = bm.asarray(kwargs["V_Na"])
        self.V_K = bm.asarray(kwargs["V_K"])
        self.V_Ca = bm.asarray(kwargs["V_Ca"])
        self.V_h = bm.asarray(kwargs["V_h"])
        self.V_l = bm.asarray(kwargs["V_l"])
        self.I_OU0 = bm.asarray(kwargs["I_OU0"])
        self.tau_OU = bm.asarray(kwargs["tau_OU"])
        self.sigma_OU = bm.asarray(kwargs["sigma_OU"])



        # Initialize state variables from kwargs
        self.V_soma = bm.Variable(bm.asarray(kwargs["V_soma_init"]))
        self.soma_k = bm.Variable(bm.asarray(kwargs["soma_k_init"]))
        self.soma_l = bm.Variable(bm.asarray(kwargs["soma_l_init"]))
        self.soma_h = bm.Variable(bm.asarray(kwargs["soma_h_init"]))
        self.soma_n = bm.Variable(bm.asarray(kwargs["soma_n_init"]))
        self.soma_x = bm.Variable(bm.asarray(kwargs["soma_x_init"]))

        self.input = bm.Variable(bm.zeros(size))  # Synaptic input (e.g. from CN)

        self.V_axon = bm.Variable(bm.asarray(kwargs["V_axon_init"]))
        self.axon_Sodium_h = bm.Variable(bm.asarray(kwargs["axon_Sodium_h_init"]))
        self.axon_Potassium_x = bm.Variable(bm.asarray(kwargs["axon_Potassium_x_init"]))

        self.V_dend = bm.Variable(bm.asarray(kwargs["V_dend_init"]))
        self.dend_Ca2Plus = bm.Variable(bm.asarray(kwargs["dend_Ca2Plus_init"]))
        self.dend_Calcium_r = bm.Variable(bm.asarray(kwargs["dend_Calcium_r_init"]))
        self.dend_Potassium_s = bm.Variable(bm.asarray(kwargs["dend_Potassium_s_init"]))
        self.dend_Hcurrent_q = bm.Variable(bm.asarray(kwargs["dend_Hcurrent_q_init"]))

        # Initialize ODE integrators - one for each variable
        self.integral_V_soma = bp.odeint(f=self.dV_soma, method=method)
        self.integral_soma_k = bp.odeint(f=self.dsoma_k, method=method)
        self.integral_soma_l = bp.odeint(f=self.dsoma_l, method=method)
        self.integral_soma_h = bp.odeint(f=self.dsoma_h, method=method)
        self.integral_soma_n = bp.odeint(f=self.dsoma_n, method=method)
        self.integral_soma_x = bp.odeint(f=self.dsoma_x, method=method)
        self.integral_V_axon = bp.odeint(f=self.dV_axon, method=method)
        self.integral_axon_Sodium_h = bp.odeint(f=self.daxon_Sodium_h, method=method)
        self.integral_axon_Potassium_x = bp.odeint(
            f=self.daxon_Potassium_x, method=method
        )
        self.integral_V_dend = bp.odeint(f=self.dV_dend, method=method)
        self.integral_dend_Ca2Plus = bp.odeint(f=self.ddend_Ca2Plus, method=method)
        self.integral_dend_Calcium_r = bp.odeint(f=self.ddend_Calcium_r, method=method)
        self.integral_dend_Potassium_s = bp.odeint(
            f=self.ddend_Potassium_s, method=method
        )
        self.integral_dend_Hcurrent_q = bp.odeint(
            f=self.ddend_Hcurrent_q, method=method
        )

        self.I_OU = bm.Variable(bm.ones(size) * self.I_OU0)  # Start at baseline
        self.I_stim = bm.Variable(bm.zeros(size)) 
        self.I_total = bm.Variable(bm.zeros(size))  # Total current including OU and stimulus

        # Spike monitoring
        self.io_threshold = kwargs["io_threshold"]
        self.spike = bm.Variable(bm.ones(self.size, dtype=bool))




    def dV_soma(
        self,
        V_soma,
        t,
        V_dend,
        V_axon,
        soma_k,
        soma_l,
        soma_h,
        soma_n,
        soma_x,
        I_OU,
        I_syn_soma,
    ):
        soma_I_leak = self.g_ls * (V_soma - self.V_l)
        I_ds = (self.g_int / self.p1) * (V_soma - V_dend)
        I_as = (self.g_int / (1 - self.p2)) * (V_soma - V_axon)
        soma_I_interact = I_ds + I_as

        soma_m_inf = 1 / (1 + bm.exp(-(V_soma + 30) / 5.5))
        soma_Ina = self.g_Na_s * soma_m_inf**3 * soma_h * (V_soma - self.V_Na)

        soma_Ikdr = self.g_Kdr_s * soma_n**4 * (V_soma - self.V_K)

        soma_Ik = self.g_K_s * soma_x**4 * (V_soma - self.V_K)

        soma_Ical = self.g_CaL * soma_k**3 * soma_l * (V_soma - self.V_Ca)

        soma_I_Channels = soma_Ik + soma_Ikdr + soma_Ina + soma_Ical
        return self.S * (
            -(soma_I_leak + soma_I_interact + soma_I_Channels) + I_OU + I_syn_soma
        )

    def dsoma_k(self, soma_k, t, V_soma):
        soma_k_inf = 1 / (1 + bm.exp(-(V_soma + 61) / 4.2))
        return soma_k_inf - soma_k

    def dsoma_l(self, soma_l, t, V_soma):
        soma_l_inf = 1 / (1 + bm.exp((V_soma + 85) / 8.5))
        soma_tau_l = (
            20 * bm.exp((V_soma + 160) / 30) / (1 + bm.exp((V_soma + 84) / 7.3))
        ) + 35
        return (soma_l_inf - soma_l) / soma_tau_l

    def dsoma_h(self, soma_h, t, V_soma):
        soma_h_inf = 1 / (1 + bm.exp((V_soma + 70) / 5.8))
        soma_tau_h = 3 * bm.exp(-(V_soma + 40) / 33)
        return (soma_h_inf - soma_h) / soma_tau_h

    def dsoma_n(self, soma_n, t, V_soma):
        soma_n_inf = 1 / (1 + bm.exp(-(V_soma + 3) / 10))
        soma_tau_n = 5 + (47 * bm.exp((V_soma + 50) / 900))
        return (soma_n_inf - soma_n) / soma_tau_n

    def dsoma_x(self, soma_x, t, V_soma):
        soma_alpha_x = 0.13 * (V_soma + 25) / (1 - bm.exp(-(V_soma + 25) / 10))
        soma_beta_x = 1.69 * bm.exp(-(V_soma + 35) / 80)
        soma_tau_x_inv = soma_alpha_x + soma_beta_x
        soma_x_inf = soma_alpha_x / soma_tau_x_inv
        return (soma_x_inf - soma_x) * soma_tau_x_inv

    def dV_axon(self, V_axon, t, V_soma, axon_Sodium_h, axon_Potassium_x):
        axon_I_leak = self.g_la * (V_axon - self.V_l)
        I_sa = (self.g_int / self.p2) * (V_axon - V_soma)
        axon_I_interact = I_sa

        axon_m_inf = 1 / (1 + bm.exp(-(V_axon + 30) / 5.5))
        axon_Ina = self.g_Na_a * axon_m_inf**3 * axon_Sodium_h * (V_axon - self.V_Na)

        axon_Ik = self.g_K_a * axon_Potassium_x**4 * (V_axon - self.V_K)

        axon_I_Channels = axon_Ina + axon_Ik
        return self.S * (-(axon_I_leak + axon_I_interact + axon_I_Channels))

    def daxon_Sodium_h(self, axon_Sodium_h, t, V_axon):
        axon_h_inf = 1 / (1 + bm.exp((V_axon + 60) / 5.8))
        axon_tau_h = 1.5 * bm.exp(-(V_axon + 40) / 33)
        return (axon_h_inf - axon_Sodium_h) / axon_tau_h

    def daxon_Potassium_x(self, axon_Potassium_x, t, V_axon):
        axon_alpha_x = 0.13 * (V_axon + 25) / (1 - bm.exp(-(V_axon + 25) / 10))
        axon_beta_x = 1.69 * bm.exp(-(V_axon + 35) / 80)
        axon_tau_x_inv = axon_alpha_x + axon_beta_x
        axon_x_inf = axon_alpha_x / axon_tau_x_inv
        return (axon_x_inf - axon_Potassium_x) * axon_tau_x_inv

    def dV_dend(
        self,
        V_dend,
        t,
        V_soma,
        dend_Calcium_r,
        dend_Potassium_s,
        dend_Hcurrent_q,
        I_app,
        I_gj,
    ):
        dend_I_leak = self.g_ld * (V_dend - self.V_l)
        dend_I_interact = (self.g_int / (1 - self.p1)) * (V_dend - V_soma)

        dend_Icah = self.g_CaH * dend_Calcium_r**2 * (V_dend - self.V_Ca)
        dend_Ikca = self.g_K_Ca * dend_Potassium_s * (V_dend - self.V_K)
        dend_Ih = self.g_h * dend_Hcurrent_q * (V_dend - self.V_h)

        I_gapp = 0.0 if I_gj is None else I_gj
        dend_I_application = -I_app - I_gapp

        dend_I_Channels = dend_Icah + dend_Ikca + dend_Ih
        return self.S * (
            -(dend_I_leak + dend_I_interact + dend_I_application + dend_I_Channels)
        )

    def ddend_Ca2Plus(self, dend_Ca2Plus, t, V_dend, dend_Calcium_r):
        dend_Icah = self.g_CaH * dend_Calcium_r**2 * (V_dend - self.V_Ca)
        return -3 * dend_Icah - 0.075 * dend_Ca2Plus

    def ddend_Calcium_r(self, dend_Calcium_r, t, V_dend):
        dend_alpha_r = 1.7 / (1 + bm.exp(-(V_dend - 5) / 13.9))
        dend_beta_r = 0.02 * (V_dend + 8.5) / (bm.exp((V_dend + 8.5) / 5) - 1.0)
        dend_tau_r_inv5 = dend_alpha_r + dend_beta_r  # tau = 5 / (alpha + beta)
        dend_r_inf = dend_alpha_r / dend_tau_r_inv5
        return (dend_r_inf - dend_Calcium_r) * dend_tau_r_inv5 * 0.2

    def ddend_Potassium_s(self, dend_Potassium_s, t, V_dend, dend_Ca2Plus):
        dend_alpha_s = bm.where(
            0.00002 * dend_Ca2Plus < 0.01, 0.00002 * dend_Ca2Plus, 0.01
        )
        dend_tau_s_inv = dend_alpha_s + 0.015
        dend_s_inf = dend_alpha_s / dend_tau_s_inv
        return (dend_s_inf - dend_Potassium_s) * dend_tau_s_inv

    def ddend_Hcurrent_q(self, dend_Hcurrent_q, t, V_dend):
        q_inf = 1 / (1 + bm.exp((V_dend + 80) / 4))
        tau_q_inv = bm.exp(-0.086 * V_dend - 14.6) + bm.exp(0.070 * V_dend - 1.87)
        return (q_inf - dend_Hcurrent_q) * tau_q_inv

    def compute_gj_currents(self, V_dend, gj_src, gj_tgt, g_gj):
        """
        Compute gap junction currents.

        Parameters
        ----------
        V_dend : array
            Dendritic membrane potentials
        gj_src : array
            Source indices for gap junctions
        gj_tgt : array
            Target indices for gap junctions
        g_gj : float
            Gap junction conductance

        Returns
        -------
        array
            Gap junction currents for each cell
        """
        vdiff = bm.subtract(bm.take(V_dend, gj_src), bm.take(V_dend, gj_tgt)) # indices
        cx36_current_per_gj = ((0.6 * bm.exp(-(vdiff * vdiff) / 2500.0) + 0.4) * vdiff * g_gj)

        I_gj = bm.zeros_like(V_dend) \
            .at[jax.numpy.array(gj_tgt)] \
            .add(cx36_current_per_gj)       # add function

        return I_gj

    def update(self, tdt=None, x=None):
        """
        Update method called by the DSRunner.

        Parameters
        ----------
        tdt : tuple, optional
            Current time and time step
        x : float or array, optional
            External current input

        Returns
        -------
        tuple
            Membrane potentials of all compartments
        """
        t = bp.share["t"]
        dt = bp.share["dt"]
        I_app = 0.0  # if x is None else x

        # --- 1. Update OU Process State ---
        xi = bm.random.normal(0, 1, self.size)
        # Using the formula: dI = (I0 - I)/tau * dt + sigma * sqrt(2/tau) * dW
        # where dW = xi * sqrt(dt)
        noise_term = self.sigma_OU * bm.sqrt(2.0 / self.tau_OU) * xi * bm.sqrt(dt)
        drift_term = (self.I_OU0 - self.I_OU) / self.tau_OU * dt
        current_I_OU = self.I_OU + drift_term + noise_term    
        current_I_total = current_I_OU + self.I_stim.value # Add external stimulus

        # --- 2. Compute Gap Junction Currents ---
        I_gj = bm.zeros(self.num)
        if hasattr(self, "gj_src") and hasattr(self, "gj_tgt"):
            I_gj = self.compute_gj_currents(
                self.V_dend, self.gj_src, self.gj_tgt, self.g_gj
            )

        # --- 3. Update Neuron States using Integrators ---
        # Get current values of state variables needed for derivatives
        V_soma = self.V_soma.value
        V_dend = self.V_dend.value
        V_axon = self.V_axon.value
        soma_k = self.soma_k.value
        soma_l = self.soma_l.value
        soma_h = self.soma_h.value
        soma_n = self.soma_n.value
        soma_x = self.soma_x.value
        axon_Sodium_h = self.axon_Sodium_h.value
        axon_Potassium_x = self.axon_Potassium_x.value
        dend_Calcium_r = self.dend_Calcium_r.value
        dend_Potassium_s = self.dend_Potassium_s.value
        dend_Hcurrent_q = self.dend_Hcurrent_q.value
        dend_Ca2Plus = self.dend_Ca2Plus.value
        I_syn_soma = self.input.value  # Get current synaptic input

        # Update soma voltage, passing the *updated* I_OU and current I_syn_soma
        new_V_soma = self.integral_V_soma(
            V_soma,
            t,
            V_dend,
            V_axon,
            soma_k,
            soma_l,
            soma_h,
            soma_n,
            soma_x,
            current_I_total,  # Pass the calculated next I_OU value
            I_syn_soma,  # Pass the current synaptic input
            dt=dt,
        )
        new_soma_k = self.integral_soma_k(soma_k, t, V_soma, dt=dt)
        new_soma_l = self.integral_soma_l(soma_l, t, V_soma, dt=dt)
        new_soma_h = self.integral_soma_h(soma_h, t, V_soma, dt=dt)
        new_soma_n = self.integral_soma_n(soma_n, t, V_soma, dt=dt)
        new_soma_x = self.integral_soma_x(soma_x, t, V_soma, dt=dt)

        new_V_axon = self.integral_V_axon(
            V_axon, t, V_soma, axon_Sodium_h, axon_Potassium_x, dt=dt
        )
        new_axon_Sodium_h = self.integral_axon_Sodium_h(axon_Sodium_h, t, V_axon, dt=dt)
        new_axon_Potassium_x = self.integral_axon_Potassium_x(
            axon_Potassium_x, t, V_axon, dt=dt
        )

        new_V_dend = self.integral_V_dend(
            V_dend,
            t,
            V_soma,
            dend_Calcium_r,
            dend_Potassium_s,
            dend_Hcurrent_q,
            0.0,  # I_app is zero for now
            I_gj,
            dt=dt,
        )
        new_dend_Ca2Plus = self.integral_dend_Ca2Plus(
            dend_Ca2Plus, t, V_dend, dend_Calcium_r, dt=dt
        )
        new_dend_Calcium_r = self.integral_dend_Calcium_r(
            dend_Calcium_r, t, V_dend, dt=dt
        )
        new_dend_Potassium_s = self.integral_dend_Potassium_s(
            dend_Potassium_s, t, V_dend, dend_Ca2Plus, dt=dt
        )
        new_dend_Hcurrent_q = self.integral_dend_Hcurrent_q(
            dend_Hcurrent_q, t, V_dend, dt=dt
        )




        # --- 4. Assign Updated Values ---
        self.I_OU.value = current_I_OU  # Assign the updated OU value
        self.I_total.value = current_I_total  # Assign the updated total current value
        self.V_soma.value = new_V_soma
        self.soma_k.value = new_soma_k
        self.soma_l.value = new_soma_l
        self.soma_h.value = new_soma_h
        self.soma_n.value = new_soma_n
        self.soma_x.value = new_soma_x

        self.V_axon.value = new_V_axon
        self.axon_Sodium_h.value = new_axon_Sodium_h
        self.axon_Potassium_x.value = new_axon_Potassium_x

        self.V_dend.value = new_V_dend
        self.dend_Ca2Plus.value = new_dend_Ca2Plus
        self.dend_Calcium_r.value = new_dend_Calcium_r
        self.dend_Potassium_s.value = new_dend_Potassium_s
        self.dend_Hcurrent_q.value = new_dend_Hcurrent_q

        # Spike tracking
        self.spike.value = self.V_soma > self.io_threshold



class IONetwork(bp.DynSysGroup):
    """
    Network of IO neurons connected by gap junctions.

    Parameters
    ----------
    num_neurons : int
        Number of neurons in the network
    g_gj : float
        Gap junction conductance
    conn_prob : callable, optional
        Probability function for connection based on distance
    rmax : int
        Maximum radius for connections in 3D grid
    nconnections : int
        Number of connections per neuron
    neuron_params : dict
        Parameters to pass to individual IONeuron instances
    """

    def __init__(
        self,
        num_neurons,
        g_gj,
        n_clusters,
        n_projections,
        p_bridge,
        **neuron_params,
    ):
        # super(IONetwork, self).__init__() 
        super().__init__()

        # Create IO neurons
        self.neurons = IONeuron(size=num_neurons, **neuron_params)


        # Set up connectivity
        src_idx, tgt_idx, n_bridges, cluster_ids=  self.sample_connections_clustered(
            num_neurons,
            n_clusters,
            n_projections,
            p_bridge)


        # Store topography params
        self.cluster_ids = cluster_ids
        self.n_bridges= n_bridges
        self.neurons.gj_src = src_idx
        self.neurons.gj_tgt = tgt_idx
        self.neurons.g_gj = g_gj



    def update(self, tdt=None, inp=None):
        """
        Update the network state.

        Parameters
        ----------
        tdt : tuple, optional
            Current time and time step
        inp : array, optional
            External input current to each neuron (not currently used by IONeuron)

        Returns
        -------
        None
            The update happens inplace within the neurons object.
        """
        # Update neurons with input
        return self.neurons()



    def sample_connections_clustered(
            self,
            nneurons,
            n_clusters = 1,
            n_projections = 4,
            p_bridge = 0,
    ):
        src_idx = []
        tgt_idx = []

        if n_clusters == 0: # Determine the clustering using a probability distribution for num neurons per cluster
            # sizes = []
            # cluster_size_probability = []
            # cluster_sizes = cluster_size_probability * np.ones(nneurons)
            pass
        else:
            base_cluster_size = nneurons // n_clusters
            remn = nneurons % n_clusters
            cluster_sizes = np.full(n_clusters, base_cluster_size, dtype=int)
            cluster_sizes[:remn] += 1

        if n_clusters == 1:
            p_bridge = 0

        assert cluster_sizes.sum() == nneurons
        assert cluster_sizes.min() - 1 > n_projections

        # Assign neurons to clusters
        perm = np.random.permutation(nneurons)
        io_idx_clustered = np.split(perm, np.cumsum(cluster_sizes)[:-1])

        cluster_ids = np.empty(nneurons, dtype=int)
        for clust_n, cluster in enumerate(io_idx_clustered):
            cluster_ids[cluster] = clust_n

        # Select bridge neurons
        if p_bridge > 0:
            while True:
                bridge_cells_idx = np.concatenate(
                    [cluster[np.random.rand(len(cluster)) < p_bridge] for cluster in io_idx_clustered])

                if len(bridge_cells_idx) < n_clusters:  # add at least 1 bridge connection for every cluster present
                    clust_pairs = np.random.choice(n_clusters, size=n_clusters,
                                                   replace=False)  # Ensure different clusters are chosen
                    new_bridge_cells = [np.random.choice(np.setdiff1d(io_idx_clustered[c], bridge_cells_idx)) for c in
                                        clust_pairs]
                    bridge_cells_idx = np.append(bridge_cells_idx, new_bridge_cells)

                if len(bridge_cells_idx) % 2 != 0:  # ensure even number of bridge cells; random draw between adding / removing cell
                    remove = np.random.rand(1) < 0.5
                    if remove:
                        to_remove = np.random.choice(bridge_cells_idx)
                        bridge_cells_idx = bridge_cells_idx[bridge_cells_idx != to_remove]
                    else:
                        bridge_cells_idx = np.append(bridge_cells_idx,
                                                     np.random.choice(np.setdiff1d(perm, bridge_cells_idx)))

                # Link cluster identity to bridge cells
                clusters = {}
                for bridge in bridge_cells_idx:
                    clust = int(cluster_ids[bridge])
                    clusters.setdefault(clust, []).append(bridge)

                # Check whether a set of bridge connections can be formed with current bridge neurons
                max_count = max(len(neurons) for neurons in clusters.values())
                if max_count <= len(bridge_cells_idx) // 2:
                    break

            # Set connections between clusters
            for c in clusters:
                np.random.shuffle(clusters[c])

            cluster_list = [(c, clusters[c]) for c in clusters]
            n_bridges = len(bridge_cells_idx) // 2

            while True:
                cluster_list = [(c, ns) for (c, ns) in cluster_list if
                                len(ns) > 0]  # Remove clusters with no remaining bridges
                if len(cluster_list) == 0:  # Stop when all bridges are paired
                    break

                # Take pairs from clusters with most amount of bridges
                cluster_list.sort(key=lambda x: len(x[1]), reverse=True)
                (c1, ns1), (c2, ns2) = cluster_list[0], cluster_list[1]

                # Select and remove a bridge from each list
                n1 = ns1.pop()
                n2 = ns2.pop()

                src_idx.extend([n1, n2])
                tgt_idx.extend([n2, n1])
        else:
            n_bridges = 0


        # Set connections within clusters
        for cluster in np.arange(n_clusters):
            src = np.concatenate([np.full(n_projections, io) for io in io_idx_clustered[cluster]])
            tgt = np.concatenate(
                [np.random.choice(np.setdiff1d(io_idx_clustered[cluster], [int(io)]), size=n_projections, replace=False)
                 for io in io_idx_clustered[cluster]])

            add_reciprocal_src = np.concatenate([np.full(((tgt == io).sum()), io) for io in io_idx_clustered[cluster]])
            add_reciprocal_tgt = np.concatenate(
                [src[np.where(tgt == io_idx_clustered[cluster][io])] for io in np.arange(cluster_sizes[cluster])])

            src = np.concatenate([src, add_reciprocal_src])
            tgt = np.concatenate([tgt, add_reciprocal_tgt])

            # Remove duplicate connections
            unique_vstack = np.unique(np.vstack((src, tgt)), axis=1)
            src = unique_vstack[0]
            tgt = unique_vstack[1]

            src_idx.extend(src.tolist())
            tgt_idx.extend(tgt.tolist())

        src_idx = bm.array(np.round(src_idx).astype(np.int32))
        tgt_idx = bm.array(np.round(tgt_idx).astype(np.int32))

        return src_idx, tgt_idx, n_bridges, cluster_ids

    def sample_connections_3d(  #
            self,
            nneurons,
            nconnections=10,
            rmax=2,  # = distance
            connection_probability=None,
            normalize_by_dr=True,
    ):

        """
        :param nneurons: int,
        Number of neurons in the network
        :param nconnections: int,
        Number of connections per neuron
        :param rmax: int,
        Maximum radius for connections in 3D grid
        :param connection_probability:
        Probability function for connection based on distance
        :param normalize_by_dr:
        Determines whether the connection profile is a continuous probability or discrete. If True,
        :return: src_idx, tgt_idx
        """
        if connection_probability is None:
            connection_probability = lambda r: np.exp(-((r / 4) ** 2))
        assert int(round(nneurons ** (1 / 3))) ** 3 == nneurons
        assert nconnections % 2 == 0
        nside = int(np.ceil(nneurons ** (1 / 3)))
        if rmax > nside / 2:
            rmax = nside // 2

        dx, dy, dz = np.mgrid[-rmax: rmax + 1, -rmax: rmax + 1, -rmax: rmax + 1]
        dx, dy, dz = dx.flatten(), dy.flatten(), dz.flatten()
        r = np.sqrt(dx * dx + dy * dy + dz * dz)  # Radius for each point in grid

        sample_backwards = (
                ((dz < 0)) | ((dz == 0) & (dy < 0)) | ((dz == 0) & (dy == 0) & (dx < 0))
        )
        m = (r != 0) & sample_backwards & (r < rmax)
        dx, dy, dz, r = dx[m], dy[m], dz[m], r[m]
        P = connection_probability(r)

        # Spread connection probability over each offset
        ro, r_uniq_idx = np.unique(r, return_inverse=True)
        r_idx_freq = np.bincount(r_uniq_idx)
        r_freq = r_idx_freq[r_uniq_idx]
        P = P / r_freq

        if normalize_by_dr:
            dr = (
                    0.5 * np.diff(ro, append=rmax)[r_uniq_idx]
                    + 0.5 * np.diff(ro, prepend=0)[r_uniq_idx]
            )
            P = P * dr
        P = P / P.sum()

        final_connection_count = nneurons * nconnections // 2

        # Decide how many edges use each offset type: for each type
        # randomly pick neurons and connect each one to its neighbor
        counts = (P * final_connection_count + 0.5).astype(int)
        counts[-1] = max(0, final_connection_count - counts[:-1].sum())
        assert (counts < nneurons).all()
        conn_idx = []
        for draw in range(len(P)):
            if counts[draw] == 0:
                continue
            if counts[draw] == 1:
                draw_idx = np.array([np.random.randint(nneurons)])
            else:
                draw_idx = np.random.choice(nneurons, counts[draw], replace=False)
            conn_idx.append(draw + len(P) * draw_idx)
        conn_idx = np.concatenate(conn_idx)

        neuron_id1 = conn_idx // len(P)

        # Coordinates of neurons in map
        x = (neuron_id1 % nside).astype("int32")
        y = ((neuron_id1 // nside) % nside).astype("int32")
        z = ((neuron_id1 // (nside * nside)) % nside).astype("int32")

        di = conn_idx % len(P)

        neuron_id2 = (
                (x + dx[di]) % nside
                + (y + dy[di]) % nside * nside
                + (z + dz[di]) % nside * nside * nside
        ).astype(int)

        tgt_idx = np.concatenate([neuron_id1, neuron_id2])
        src_idx = np.concatenate([neuron_id2, neuron_id1])

        # Convert to BrainPy indices
        src_idx = bm.array(src_idx, dtype=bm.uint32)
        tgt_idx = bm.array(tgt_idx, dtype=bm.uint32)
        
        return src_idx, tgt_idx


