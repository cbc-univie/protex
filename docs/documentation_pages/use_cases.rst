.. _Use_Cases:

Use Cases
=========

A collection of some useful situation for PROTEX.

HPTS
----

The photoacid 8-Hydroxypyrene-1,3,6-trisulfonic acid (HPTSH) can be used to introduce an excess proton into the system. It may be insightful to investigate the transfers this proton undergoes during the simulation.

The protonated (HPTSH) and deprotonated (HPTS) forms of the molecule need to be set up with `CHARMM-GUI <https://www.charmm-gui.org/>`_ and Drude parametrized with `FFParam <http://ffparam.umaryland.edu/>`_, using dummy atoms and dummy lone pairs where necessary.
``protex`` can be set up as described in the :ref:`Quick-Start-Guide`, with the exception that the new species need to be added to the templates.
The deprotonated acid shouldn't accept any protons, so only the reactions involving the protonated form should be added to the allowed updates:

.. code-block:: python

    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.16, "prob": 0.994}
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.16, "prob": 0.098}
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.16, "prob": 0.201}
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.16, "prob": 0.684}
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.16, "prob": 1.0}
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.16, "prob": 1.0}

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1, HPTS_HPTSH], allowed_updates)
    ionic_liquid = IonicLiquidSystem(simulation, templates)

Carry out simulations as described before.

To analyse the diffusion of the extra proton, iterate through the updates and check whether the molecule carrying the proton underwent a proton transfer. In the end you should have a list of the residue index of the molecule carrying the proton for each time step.
You can use this list to index a selection of H atoms with `MDAnalysis <https://www.mdanalysis.org/>`_ and find the position of the proton in the trajectories.

.. code-block:: python

    u = MDAnalysis.Universe(psf, dcd)
    sel_h = u.select_atoms("(resname IM1H and name H7) or (resname IM1 and name H7) or (resname OAC and name H) or (resname HOAC and name H) or (resname HPTS and name H7) or (resname HPTSH and name H7)")
    (...)
    ctr = 0
    for ts, step, idx in zip(u.trajectory[::skip], list(charge_steps)[::skip], h_idx):
        pos_h[ctr] = sel_h[idx].position
        ctr += 1

Then calculate the MSD using these coordinates.
