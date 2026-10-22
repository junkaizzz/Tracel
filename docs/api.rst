API documentation
=================

.. automodule:: tracel
   :no-members:

.. currentmodule:: tracel

Cell alignment (OT1)
--------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   run_ot1
   OT1Params
   OT1Result
   log_normalize

Transcript alignment (OT2)
--------------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   prepare_ot2_inputs
   make_windows
   run_ot2_windows
   solve_window
   merge_ot2
   OT2Params

Lower-level helpers
-------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   ot1.type_balanced_marginal
   ot1.structure_embedding
   ot1.expression_embedding
   ot1.lowrank_blend
   ot2.select_winners
   ot2.threshold_decisions
