.. _Quick-Start-Guide:

Quick Start Guide
=================

Here is how to easily set up your system and start with Protex!

``protex`` works together with `OpenMM <https://openmm.org>`_ to allow bond breaking and formation (i.e. for a proton transfer) during MD Simulations.
First, obtain an OpenMM ``simulation`` object (there are helper functions provided by protex- implement!).
Afterwards the main pathway is to specifiy the allowed transfers using ``IonicLiquidTemplates`` and wrap the simulation and templates into an ``IonicLiquidSystem``.

.. code-block:: python

    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.994}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.098}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], allowed_updates)
    ionic_liquid = IonicLiquidSystem(simulation, tempaltes)

Then define the update method which suits you best (currently only monte carlo update based on distance and prob available TODO -> fix and sort). 

.. code-block:: python

    update = NaiveMCUpdate(ionic_liquid)
    state_update = StateUpdate(Update)

Now you are ready to run the simulation and just call the update method whenever you like.

.. code-block:: python

    ionic_liquid.simulation.step(1000)
    state_update.update(2)

There is also a ``ChargeReporter``.