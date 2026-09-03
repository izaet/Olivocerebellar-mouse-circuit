--> models --> mouse
- mouse-free.xml
          Describes the joints of the mouse model
- eigenmode_mjx.py
          Describes how the angles of these joints are translated to Brainpy math variables and the other way around
          Creates Body as a Brainpy Neural Dynamics class 
- network_body.py
          Describes SynConn class AnglesToPC to meaningfully connect the mouse model's joint angles to the Purkinje cell population.
          Contains Mouse as a Brainpy Dynamic system class in which both the Olivocerebellar circuit and the Mujoco MJX mouse model are incorporated.
