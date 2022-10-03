.. _Detailed-Example:

Detailed Example
=================

**!!!Under Construction!!!**

.. admonition:: |:confetti_ball:| Congratulations! |:confetti_ball:|
   :class: successstyle

   It seems you realy want to learn how to use protex! 
   Keep going! |:smile:|

Setup
-----

Now we will dive a little bit deeper in all of the functions and methods protex offers. 
First and foremost we need to make sure to have valid input files!
The usual way is to provide a psf and crd file alongside the toppar files. 
The topology and parameter files (toppar) contain all the residues needed for the system setup.
It is really important, that the corresponding residues prone to (de-)protonation have the same number of atoms in the same order.
So, i.e if we have methylimidazolium (im1h) and methylimidazole (im1) the missing hydrogen for im1 needs to be a dummy atom. The same is true for acetate/acetic acid.
Additionally, make sure that also atoms in the coordinate files match the residue description. 

.. attention:: 
    It is VERY important that the atom order of the protonated and deprotonated residues match exactly between the coordinate as well as topology/psf files!

Now, lets start building our system. The first part, getting the OpenMM simulaiton object is nothing protex specific and we won't need any protex functions for it. 
Nevertheless, here is one possible way to do it, if you are not that familiar with OpenMM.

.. note:: 
    All the files we need can be found in the protex directory. 

.. code-block:: python

    from openmm import Platform
    from openmm.app import PME, CharmmCrdFile, CharmmParameterSet, CharmmCrdFile, Simulation
    from openmm.unit import angstroms, kelvin, picoseconds
    from velocityverletplugin import VVIntegrator

    protex_dir = "/path/to/your/protex/directory"

    print("Loading CHARMM files...")
    PARA_FILES = ["toppar_drude_master_protein_2013f_lj025.str", "hoac_d.str", "im1h_d.str", "im1_d.str", "oac_d.str"]
    params = CharmmParameterSet(*[f"{protex_dir}/forcefield/toppar/{para_files}" for para_files in PARA_FILES])
    psf = CharmmPsfFile(f"{protex_dir}/forcefield/im1h_oac_150_im1_hoac_350.psf")
    psf.setBox(48*angstroms, 48*angstroms, 48*angstroms)
    crd = CharmmCrdFile(f"{protex_dir}/forcefield/im1h_oac_150_im1_hoac_350.crd")

    print("Create system")
    system = psf.createSystem(params, nonbondedMethod=PME, nonbondedCutoff=11.0 * angstroms, switchDistance=10 * angstroms, constraints=None)

    print("Setup simulation")
    integrator = VVIntegrator(300 * kelvin, 10 / picoseconds, 1 * kelvin, 100 / picoseconds, 0.0005 * picoseconds)
    integrator.setMaxDrudeDistance(0.25 * angstroms)
    platform = Platform.getPlatformByName("CUDA")
    prop = dict(CudaPrecision="single")
    simulation = Simulation(psf.topology, system, integrator, platform=platform, platformProperties=prop)
    simulation.context.setPositions(crd.positions)

Now we have the simulation object ready. In principle we did, what was done with ``generate_im1h_oac_system()``. For advanced usage of this function look :ref:`Advanced setup`

The next thing is to get everything ready for the ``IonicLiquidTemplates`` class, which we will need beside the simulation to build the ``IonicLiquidSystem``.

    

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

.. _Advanced Setup:

Advanced Setup
--------------

.. code-block:: python

    from protex.testsystems import generate_im1h_oac_system

    simulation = generate_im1h_oac_system()

restart file, psf file, ....