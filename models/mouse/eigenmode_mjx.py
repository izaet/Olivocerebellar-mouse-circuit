import os
# os.environ['JAX_PLATFORMS'] = 'cpu'
import brainpy as bp
import brainpy.math as bm
import mujoco
from mujoco import mjx
import jax
from jax import numpy as jnp
from types import SimpleNamespace
import numpy as np

# from brax.io import mjcf
# from brax.io import html
# from brax.generalized import pipeline



class Body(bp.dyn.NeuDyn):
    def __init__(self, delta=None):
        super().__init__(size=24)
        try:
            mm_model = mujoco.MjModel.from_xml_path('mouse-free.xml')  # model in mjx
        except:
            path = os.path.join(os.path.dirname(__file__), 'mouse-free.xml')
            mm_model = mujoco.MjModel.from_xml_path(path)
            
        self.sys = mjx.put_model(mm_model)
        self.mj_model = mm_model
        angle0 = jnp.array([-60., 110., 60.,]*2 + [60., -100., -30.]*2)
        if delta is None:
            delta  = 10 * jnp.array([-1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1])
        else:
            delta  = 10 * jnp.array(delta)

        data= mujoco.MjData(mm_model)
        data.qpos = jnp.pi/180*(angle0 + delta)
        data.qvel = jnp.zeros_like(data.qpos)
        state_mjx = mjx.put_data(mm_model, data) #bm.Variable(mjx.put_data(mm_model, data))
        self._dataclass = type(state_mjx)
        self.state = bm.Variable({name: getattr(state_mjx, name) for name in (state_mjx.__dataclass_fields__.keys())})



    @property
    def angle(self):
        return self.state_mjx.qpos * 180 / jnp.pi # type: ignore

    @property
    def output(self):
        angle0 = jnp.array([-60., 110., 60.,]*2 + [60., -100., -30.]*2)
        x = angle0 - self.angle
        out = 0.1 * jnp.concatenate([
            #jax.nn.relu(+ x),
            #jax.nn.relu(- x)
            x,
            ])
        return out

    @property
    def state_mjx(self):
        return self._dataclass(**self.state.value)

    def update(self):
        dt = bp.share['dt'] * 1e-3 # ms to s
        self.dt = bp.share['dt']
        sys = self.sys.replace(opt=self.sys.opt.replace(timestep=dt))
        angle0 = jnp.array([-60., 110., 60.,]*2 + [60., -100., -30.]*2)
        action = 1e-6 * -(self.angle-angle0)
        state_mjx = self.state_mjx
        state_mjx = mjx.step(self.sys, state_mjx.replace(ctrl=action))

        for name in self.state_mjx.__dataclass_fields__.keys():
            self.state.value[name] = getattr(state_mjx, name)

        return self.state
    
    def render(self, mon, fn='index.html', height=840, js='', subsample=1):
        states = [jax.tree_util.tree_map(lambda x: x[i], mon) for i in range(mon['qpos'].shape[0])][::subsample]
        if any(jnp.isnan(states[-1]['qpos'])):
            states_nonnan = [x for x in states if not any(jnp.isnan(x['qpos']))]
            if len(states) != len(states_nonnan):
                print('nanstates, only showing', len(states_nonnan), 'out of', len(states))
            states = states_nonnan

        with open(fn, 'w') as f:
            sys = self.sys.replace(opt=self.sys.opt.replace(timestep=self.dt*subsample))



            doc = html.render(sys, states, height=height)
            doc = doc.replace('var viewer = new Viewer(domElement, system);',
                              'var viewer = new Viewer(domElement, system); document.viewer=viewer; ' + js)
            print(doc, file=f)

 
# class AngleProprioception(bp.dyn.NeuDyn):
#    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
#        assert isinstance(pre, Body)
#        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
#    def update(self):
#        self.post.I_PC.value += total_increments

if __name__ == '__main__':
    duration = 2000
    dt = 1.0
    net = Body()
    monitors = {
        'body': net.state
            }
    runner = bp.DSRunner(net, monitors=monitors, dt=dt)
    runner.progress_bar = False
    runner._fun_predict = bm.jit(runner._fun_predict)
    runner.run(duration)
    mon = runner.mon['body']
    # net.render(runner.mon['body'])
