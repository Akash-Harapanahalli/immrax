Parametric Embeddings
=====================

Reachable sets as parametopes (polytopes, normotopes, Chebyshev polytopes)
evolved by adjoint embeddings (``examples/parametric/``).

ReachiLQR
---------

Iterative-LQR synthesis of tight, certified reachable tubes.

.. toctree::
   :maxdepth: 1

   examples/parametric/iLQR/vdp
   examples/parametric/iLQR/vdp_poly
   examples/parametric/iLQR/robot_arm

Chebyshev polytope: Van der Pol
-------------------------------

ARCH-COMP Van der Pol (``mu = 1``), spec ``y < 2.75`` up to ``t = 7``, with
residual bounds on a cell partition.

.. image:: examples/parametric/chebyshev_polytope/vdp.png

.. literalinclude:: examples/parametric/chebyshev_polytope/vdp.py
   :language: python

Chebyshev polytope: coupled Van der Pol
---------------------------------------

ARCH-COMP coupled Van der Pol (4 states), with every bound a validated SOS
support function over the set.

.. image:: examples/parametric/chebyshev_polytope/coupled_vdp.png

.. literalinclude:: examples/parametric/chebyshev_polytope/coupled_vdp.py
   :language: python

Reverse Van der Pol
-------------------

Four certified methods from the same box, with the bloat over the true set.

.. image:: examples/parametric/reverse_vdp/reverse_vdp.svg

.. image:: examples/parametric/reverse_vdp/reverse_vdp_bloat.svg

.. literalinclude:: examples/parametric/reverse_vdp/reverse_vdp.py
   :language: python

Robot arm
---------

.. image:: examples/parametric/robot_arm/robot_arm.svg

.. literalinclude:: examples/parametric/robot_arm/robot_arm.py
   :language: python

Symplectic vs. non-symplectic integration
-----------------------------------------

.. image:: examples/parametric/symplectic_drift/symplectic_drift_oscillator.svg

.. literalinclude:: examples/parametric/symplectic_drift/symplectic_drift.py
   :language: python
