import brainpy as bp
import brainpy.math as bm

class HalfWaveStimIOPF(bp.dyn.NeuDyn):
    def __init__(self, n_io, n_pf, **kwargs):
        """
        o Give half waves
       Args: 
       n_io: Number of IO cells
       n_pf: Number of PFs

       kwargs: Dictionary containing the following parameters:
        isi: Interval or overlap between the input to the inferior olive and parallel fibre stimuli (ms)  t_PF - t_IO
        stim_freq: How often stimuli occur (per ms)
        dur_io : Duration of the IO half-wave (ms)
        dur_pf : Duration of the PF half-wave (ms)
        amp_io:
        amp_pf:
        phase_io:
        phase_pf:


        # Order and timing of PF and IO stimuli depending on ISI
        # Positive ISI: PF leads IO by ISI ms
        # with overlap if ISI < dur_PF
        # Negative ISI: IO leads PF by |ISI| ms
        # with overlap if |ISI| < dur_IO
        # Zero ISI: simultaneous onset
        """
        super().__init__(size = n_io+n_pf)

        self.n_io = n_io
        self.n_pf = n_pf

        # Turning stimulus off/ on
        self.stim_io_on = bm.Variable(bm.asarray(kwargs["OU_stim_io_on"]), dtype=bool)
        self.stim_pf_on = bm.Variable(bm.asarray(kwargs["OU_stim_pf_on"]), dtype=bool)
       

        # Stimulus timing
        self.isi_mean = bm.Variable(bm.array(kwargs['OU_stim_isi_mean']))
        self.isi_std = bm.Variable(bm.array(kwargs['OU_stim_isi_std']))
        self.current_isi = bm.Variable(bm.random.normal(self.isi_mean, self.isi_std))

        self.stim_freq = kwargs['OU_stim_freq']
        self.t_stim_next = bm.Variable(bm.array(float(kwargs["OU_stim_start"])), dtype=bm.float32)
        self.stim_pf_start = bm.Variable(bm.array(0.0))
        self.stim_io_start = bm.Variable(bm.array(0.0))

        self.in_stim = bm.Variable(bm.array(False))

        # Stimulus characteristic parameters
        self.dur_io = kwargs["OU_stim_dur_io_mean"]
        self.dur_pf = kwargs["OU_stim_dur_pf_mean"]
        self.amp_io = kwargs["OU_stim_amp_io_mean"]
        self.amp_pf = kwargs["OU_stim_amp_pf_mean"]

        # Masks for IO and PF selection
        self.io_mask = bm.Variable(bm.ones(self.n_io, dtype= bool))
        self.pf_mask = bm.Variable(bm.ones(self.n_pf, dtype = bool))

        # Output containing half waves
        self.M_io = bm.Variable(bm.zeros(self.n_io)) 
        self.M_pf = bm.Variable(bm.zeros(self.n_pf))


    def update(self):

    

        t = bp.share["t"]

        # Order and timing of PF and IO stimuli depending on ISI
        pf_leads = self.current_isi.value > 0.0

        self.stim_pf_start.value = bm.where(pf_leads, self.t_stim_next.value, self.t_stim_next.value + bm.abs(self.current_isi.value),)

        self.stim_io_start.value = bm.where(pf_leads,self.t_stim_next.value + bm.abs(self.current_isi.value),self.t_stim_next.value,)

        stim_io_end = self.stim_io_start.value + self.dur_io
        stim_pf_end=  self.stim_pf_start.value + self.dur_pf
        stim_end = bm.maximum(stim_io_end, stim_pf_end) 

        # Update states
        stim_start = (t >= self.t_stim_next.value) & (~self.in_stim.value)
        self.in_stim.value = bm.where(stim_start, True, self.in_stim.value)

        in_stim_io = ( t >= self.stim_io_start.value ) & ( t < stim_io_end) & (self.in_stim.value)
        in_stim_pf = ( t>= self.stim_pf_start.value  ) & ( t < stim_pf_end) & (self.in_stim.value)
        
        # Base signal
        M_io_scalar = bm.where(in_stim_io , self.amp_io * bm.sin(bm.pi * ((t-self.stim_io_start.value)/ self.dur_io)), 0)
        M_pf_scalar = bm.where(in_stim_pf, self.amp_pf * bm.sin(bm.pi * ((t-self.stim_pf_start.value)/ self.dur_pf)), 0)

        # Mask to apply stimuli to specific IO and PF neurons, vectorization
        M_io_vec = bm.full_like(self.M_io, M_io_scalar)
        M_pf_vec =  bm.full_like(self.M_pf, M_pf_scalar)

        self.M_io.value =  bm.where(self.stim_io_on.value & self.io_mask,M_io_vec,0.0,)
        self.M_pf.value = bm.where(self.stim_pf_on.value & self.pf_mask, M_pf_vec, 0.0)

        # Update timing values & ISI for next stimulus
        stim_finish = (t >= stim_end) & (self.in_stim.value)

        self.in_stim.value = bm.where(stim_finish, False, self.in_stim.value)
        self.t_stim_next.value =  bm.where(stim_finish, stim_end + self.stim_freq, self.t_stim_next.value)
        self.current_isi.value  = bm.where(stim_finish, bm.random.normal(self.isi_mean, self.isi_std), self.current_isi.value)




        