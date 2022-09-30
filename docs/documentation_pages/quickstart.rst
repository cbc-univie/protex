.. _Quick-Start-Guide:

Quick Start Guide
=================

Here is how to easily set up your system and start with protex!

``protex`` works together with `OpenMM <https://openmm.org>`_ to allow bond breaking and formation (i.e. for a proton transfer) during MD Simulations.
First, obtain an OpenMM ``simulation`` object. For purposes of this tutorial we will use a helper function.

.. code-block:: python

    from protex.testsystems import generate_im1h_oac_system

    simulation = generate_im1h_oac_system()
    

.. attention:: 
    It is VERY important that the atom order of the protonated and deprotonated residues match exactly between the coordinate as well as topology/psf files!

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

    infos={"Put whatever additional infos you would like the charge reporter to store here"}
    save_freq = 200
    charge_reporter = ChargeReporter(f"path/to/outfile", save_freq, ionic_liquid, header_data=infos)
    ionic_liquid.simulation.reporters.append(charge_reporter)

You can add additional OpenMM reporters:

.. code-block:: python

    from openmm.reporters import ..


Now you are ready to run the simulation and just call the update method whenever you like.

.. code-block:: python

    ionic_liquid.simulation.step(1000)
    state_update.update(2)
