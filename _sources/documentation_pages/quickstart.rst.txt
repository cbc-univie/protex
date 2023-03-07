.. _Quick-Start-Guide:

Quick Start Guide
=================

Here is how to easily set up your system and start with protex!

.. important::  
    If not already done, use the :ref:`installation instructions <installation-instructions>` to get protex.
    For purposes of this tutorial it is important to have the :ref:`VVIntegrator plugin <install using conda env>` installed!

``protex`` works together with `OpenMM <https://openmm.org>`_ to allow bond breaking and formation (i.e. for a proton transfer) during MD Simulations.
First, obtain an OpenMM ``simulation`` object. For purposes of this tutorial we will use a helper function.

.. code-block:: python

    from protex.testsystems import generate_im1h_oac_system

    simulation = generate_im1h_oac_system()
    
Afterwards the main pathway is to specifiy the allowed transfers and which atoms are subject to the transfer using ``IonicLiquidTemplates``. 
Then wrap the simulation and templates into an ``IonicLiquidSystem``.

.. code-block:: python

    from protex.system import IonicLiquidSystem, IonicLiquidTemplates

    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.994}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.098}

    IM1H_IM1 = {"IM1H": {"atom_name": "H7", "canonical_name": "IM1"},
                 "IM1": {"atom_name": "N2", "canonical_name": "IM1"}}

    OAC_HOAC = {"OAC" : {"atom_name": "O2", "canonical_name": "OAC"},
                "HOAC": {"atom_name": "H", "canonical_name": "OAC"}}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1], allowed_updates)
    ionic_liquid = IonicLiquidSystem(simulation, templates)


Next define the update method. 

.. code-block:: python

    from protex.update import NaiveMCUpdate, StateUpdate

    update = NaiveMCUpdate(ionic_liquid)
    state_update = StateUpdate(update)

Optionally you can define reporters for the simulation. Protex has a built in ``ChargeReporter`` to report the current charges of all molecules which can just be added to the simulation like all other OpenMM reporters.

.. code-block:: python

    from protex.reporter import ChargeReporter

    save_freq = 200
    charge_reporter = ChargeReporter(f"path/to/outfile", save_freq, ionic_liquid)
    ionic_liquid.simulation.reporters.append(charge_reporter)

You can add additional OpenMM reporters to the ionic liquid object:

.. code-block:: python

    from openmm.app import StateDataReporter, DCDReporter
    report_frequency = 200
    ionic_liquid.simulation.reporters.append(DCDReporter(f"traj.dcd", report_frequency))
    state_data_reporter= StateDataReporter(sys.stdout,
        report_frequency,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=False,
    )
    ionic_liquid.simulation.reporters.append(state_data_reporter)

Now you are ready to run the simulation and just call the update method whenever you like.

.. code-block:: python

    for i in range(10):
        ionic_liquid.simulation.step(1000)
        state_update.update(2)

.. admonition:: |:confetti_ball:| Congratulations! |:confetti_ball:|
   :class: successstyle

   You ran your first protex simulation and manged to break and build bonds during an MD Simulation!