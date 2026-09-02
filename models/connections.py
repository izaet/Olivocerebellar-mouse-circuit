class PFtoPC_BCM(bp.dyn.SynDyn):
    def __init__(self, pre, post, pre_idx, post_idx,  size=None,keep_size=False, sharding=None,name=None,mode=None,method='exp_auto', **kwargs):
        super().__init__(size=size, keep_size=keep_size, sharding=sharding, name=name, mode=mode, method=method)

        self.plasticity_on = bm.Variable(bm.asarray(kwargs["plasticity_on"]), dtype=bool)

        self.pre = pre
        self.post = post
        self.pre_idx = pre_idx
        self.post_idx = post_idx
        self.num_connections = size

        if self.num_connections != len(kwargs["init_weights"]):
            raise ValueError(
                "PFtoPC connection error: weights per connection and num connection length mismatch."
            )

        # Variables
        self.weights_per_conn = bm.Variable( bm.asarray(kwargs["init_weights"]))
        self.w_cspk = bm.Variable(bm.zeros(self.num_connections)) # Initial value of PF_PC weight is completely determined by BCM/LTP weights
        self.w_BCM = bm.Variable(bm.zeros(self.num_connections))# bm.Variable(bm.asarray(kwargs["init_weights"]))
        self.theta_M = bm.Variable(bm.ones(self.post.size) * (kwargs["theta_M_init"])) # Plasticity threshold stored
        self.init_weight = kwargs["init_weights"]


        self.dw_cspk = bm.Variable(bm.zeros(self.num_connections))
        self.dw_BCM = bm.Variable(bm.zeros(self.num_connections))


        # Parameters
        self.A_cspk = kwargs["A_cspk"]
        self.tau_cspk = kwargs["tau_cspk"]
        self.tau_M = kwargs["tau_M"]
        self.I_PF_0 = kwargs["I_PF_0"]
        self.pf_scaling = kwargs["pf_scaling"]


    def update(self):
        
        dt = bp.share["dt"]

        # Scale final weight change when plasticity == off
        plasticity_gate= self.plasticity_on.value.astype(bm.float32)

        # Firing rates
        rho_PF= bm.asarray(self.pre.rho.value)
        rho_PC = bm.asarray(self.post.rho.value)

        # Update sliding plasticity threshold
        self.theta_M.value = self.theta_M.value* bm.exp(-dt / self.tau_M)  + self.tau_M * bm.square(rho_PC) * (1.0 - bm.exp(-dt / self.tau_M))   # BCM sliding threshold (Eq. 16)

        # Convert values to per connection
        theta_M_per_con = bm.take(self.theta_M.value, self.post_idx)
        rho_PC_per_con = bm.take(rho_PC, self.post_idx)
        rho_PF_per_con = bm.take(rho_PF, self.pre_idx)
        cspk_per_con = bm.take(self.post.cspk.value, self.post_idx)  # Boolean for connections where PC has cspk

        # BCM / LTP rule
        self.dw_BCM.value = dt * (self.pf_scaling *rho_PF_per_con * bm.tanh(10.0* ( ( rho_PC_per_con*( rho_PC_per_con-theta_M_per_con)) /theta_M_per_con))) # (Eq. 18)
        self.w_BCM.value += self.dw_BCM.value

        # CSpk / LTD rule
        dw_cspk_increase_all = (self.A_cspk * bm.abs(rho_PF_per_con - self.I_PF_0))
        dw_cspk_increase = bm.where(cspk_per_con, dw_cspk_increase_all, 0.0) # LTD increase for synapses with cspk (Eq. 20)
        dw_continuous = -self.w_cspk.value / self.tau_cspk * dt # Continuous decay of LTD for all synapses (Eq. 19)

        self.dw_cspk.value = dw_continuous + dw_cspk_increase
        self.w_cspk.value = self.w_cspk.value * bm.exp(-dt / self.tau_cspk) + dw_cspk_increase

        # Final sum of weights
        self.weights_per_conn.value = bm.clip(self.init_weight + (plasticity_gate * 0.4 * (self.w_BCM.value + self.w_cspk.value)), 0 ,5) # (Eq. 14)


class PFtoPC(bp.dyn.SynConn):
    def __init__(self, pre, post, syndyn, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
        self.add = kwargs.get('add', False)

        self.pre = pre
        self.post = post
        self.conn = conn
        self.syndyn = syndyn

        self.pre_indices_flat = self.conn.require(
            "pre_ids"
        )  # shape: (num_connections,)
        self.post_indices_flat = self.conn.require(
            "post_ids"
        )  # shape: (num_connections,)
        self.num_connections = len(self.pre_indices_flat)

        if self.num_connections == 0:
            # Warning handled, no need to comment
            pass
        if len(self.pre_indices_flat) != len(self.post_indices_flat):
            raise ValueError(
                "PFtoPC connection error: pre_ids and post_ids length mismatch."
            )

    def update(self):
        pre_I = self.pre.I_OU.value  # shape: (num_pf,)
        weights_per_conn = self.syndyn.weights_per_conn.value

        if self.syndyn.num_connections == 0:
            self.post.input = bm.zeros(self.post.num)  # shape: (num_pc,)
            return

        # Use stored flat indices
        pre_I_per_conn = bm.take(
            pre_I, self.pre_indices_flat
        )  # shape: (num_connections,)
         # shape: (num_connections,)
        contribution_per_conn = (
            (1 / 5.0) * weights_per_conn * pre_I_per_conn
        )  # shape: (num_connections,)

        # Sum contributions using segment_sum
        total_input = bm.segment_sum(
            contribution_per_conn,
            self.post_indices_flat,  # Segment IDs
            num_segments=self.post.num,
        )  # Output shape: (num_pc,)
        if self.add:
            self.post.input.value += total_input  # shape: (num_pc,)
        else:
            self.post.input.value = total_input  # shape: (num_pc,)

class PCToCN(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.gamma_PC = kwargs["gamma_PC"]
        self.delay = kwargs["delay"]
        # indices, indptr for pre->post mapping
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        # self.post_indices shape: (num_connections,)
        # self.post_indptr shape: (num_pre + 1,)
        self.delay_length = int(self.delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)

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

    def update(self):
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_pre,) Boolean

        # Check which connections originated from a spiking neuron
        source_spiked_mask = bm.take(
            delayed_spikes, self.source_indices_per_conn
        )  # shape: (num_connections,) Boolean
        # Calculate increments (gamma_PC or 0) for each connection
        connection_increments = bm.where(
            source_spiked_mask, self.gamma_PC, 0.0
        )  # shape: (num_connections,)

        # Sum increments for each target postsynaptic neuron
        total_increments = bm.segment_sum(
            connection_increments,
            self.post_indices,  # Target indices as Segment IDs
            num_segments=self.post.num,  # Output shape: (num_post,) or (num_cn,)
        )

        self.post.I_PC.value += total_increments


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


class IOToPC(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.cs_weight = kwargs["cs_weight"]
        self.io_threshold = kwargs["io_threshold"]
        self.delay = kwargs["delay"]
        self.delay_length = int(self.delay / bp.share["dt"])

        self.spike_delay = bm.LengthDelay(
            self.pre.V_soma > self.io_threshold,
            self.delay_length + 1,
        )

        # indices, indptr for post->pre mapping (PC -> its single IO source)
        (self.io_source_indices, self.pc_target_indptr) = self.conn.require("post2pre")
        # self.io_source_indices shape: (num_connections,) or (num_pc,)
        # self.pc_target_indptr shape: (num_post + 1,) or (num_pc + 1,)
        if len(self.io_source_indices) != post.num:
            raise ValueError("IO->PC connection error: Expected one IO source per PC.")

        self.last_w_increment = bm.Variable(bm.zeros(post.num))  # shape: (num_pc,)



    def update(self):
        self.spike_delay.update(self.pre.V_soma > self.io_threshold)

        spiked_now_delayed = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_io,)
        spiked_pre_delayed = self.spike_delay.retrieve(
            self.delay_length + 1
        )  # shape: (num_io,)

        # Detect threshold crossing (rising edge)
        rising_edge_delayed = (
            spiked_now_delayed & ~spiked_pre_delayed
        )  # shape: (num_io,) Boolean

        io_source_rising_edge = bm.take(
            rising_edge_delayed, self.io_source_indices
        )  # shape: (num_pc,) Boolean

        # Calculate w increment only on rising edge
        w_increment = bm.where(
            io_source_rising_edge, self.cs_weight, 0.0
        )  # shape: (num_pc,)
        self.last_w_increment.value = w_increment  # Store for monitoring
        self.post.w.value += self.last_w_increment

        # Track complex spikes
        self.post.cspk.value = io_source_rising_edge  # Boolean for each PC cell

class stimToIO (bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
        (self.stim_source_indices, self.io_target_indptr) = self.conn.require("post2pre")
        self.pre = pre
        self.post = post

    def update(self):
        self.post.I_stim.value = self.pre.M_io.value


class stimToPF (bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
        (self.stim_source_indices, self.pf_target_indptr) = self.conn.require("post2pre")
        self.pre = pre
        self.post = post

    def update(self):
         self.post.I_stim.value = self.pre.M_pf.value   
