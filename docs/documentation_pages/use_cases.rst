.. _Use_Cases:

Use Cases
=========

A collection of some useful situation for PROTEX.

HPTS
----

The photoacid 8-Hydroxypyrene-1,3,6-trisulfonic acid (HPTSH) can be used to introduce an excess proton into the system. It may be insightful to investigate the transfers this proton undergoes during the simulation. In order to compare the results with experimental data, the system needs to be set up in methanol.

The protonated (HPTSH) and deprotonated (HPTS) forms of the molecule needed to be set up with `CHARMM-GUI <https://www.charmm-gui.org/>`_ and Drude parametrized with `FFParam <http://ffparam.umaryland.edu/>`_, using dummy atoms and dummy lone pairs where necessary. The protonated methanol ("MEOH2") was also set up in this manner. These files can now be found in the protex/forcefield directory.

``protex`` can be set up as described in the :ref:`Quick-Start-Guide`, with the exception that the new species need to be added to the templates.
It is still under discussion, whether the deprotonated acid should accept any protons, so take care whether you want to include reactions involving the deprotonated form in the allowed updates. In either case, we want to ensure that the acid is deprotonated as soon as possible, so the probability of the reactions of HPTSH with any partner have to be set to 1.

.. code-block:: python

    HPTSH_HPTS = {"HPTSH" : {"atom_name": "H7", "canonical_name": "HPTS"},
            "HPTS": {"atom_name": "O7", "canonical_name": "HPTS"}}

    MEOH_MEOH2 = {"MEOH": {"atom_name": "O1","canonical_name": "MEOH"},
            "MEOH2": {"atom_name": "HO2","canonical_name": "MEOH"}}
    
    allowed_updates = {}
    allowed_updates[frozenset(["IM1H", "OAC"])] = {"r_max": 0.155, "prob": 0.994}  
    allowed_updates[frozenset(["IM1", "HOAC"])] = {"r_max": 0.155, "prob": 0.098} 
    allowed_updates[frozenset(["IM1H", "IM1"])] = {"r_max": 0.155, "prob": 0.201} 
    allowed_updates[frozenset(["HOAC", "OAC"])] = {"r_max": 0.155, "prob": 0.684} 
    allowed_updates[frozenset(["HPTSH", "OAC"])] = {"r_max": 0.155, "prob": 1.000} 
    allowed_updates[frozenset(["HPTSH", "IM1"])] = {"r_max": 0.155, "prob": 1.000}
    #allowed_updates[frozenset(["HPTSH", "HPTS"])] = {"r_max": 0.155, "prob": 1.000} 
    #allowed_updates[frozenset(["HOAC", "HPTS"])] = {"r_max": 0.155, "prob": 1.000} 
    #allowed_updates[frozenset(["IM1H", "HPTS"])] = {"r_max": 0.155, "prob": 1.000}
    allowed_updates[frozenset(["HPTSH", "MEOH"])] = {"r_max": 0.155, "prob": 1.000} 
    allowed_updates[frozenset(["MEOH2", "MEOH"])] = {"r_max": 0.155, "prob": 1.000} 
    allowed_updates[frozenset(["MEOH2", "IM1"])] = {"r_max": 0.155, "prob": 1.000} 
    allowed_updates[frozenset(["MEOH2", "OAC"])] = {"r_max": 0.155, "prob": 1.000} 
    

    templates = IonicLiquidTemplates([OAC_HOAC, IM1H_IM1, HPTSH_HPTS, MEOH_MEOH2], allowed_updates)
    ionic_liquid = IonicLiquidSystem(simulation, templates)

Carry out simulations as described before.

To analyse the diffusion of the extra protons, iterate through the updates and check whether the molecules carrying the protons underwent a proton transfer. In the end you should have a list of the residue index of the molecule carrying the proton for each time step and each proton that originally came from an HPTSH.
You can use this list to index a selection of H atoms with `MDAnalysis <https://www.mdanalysis.org/>`_ and find the position of the protons while looping through the trajectories.

.. code-block:: python

    u = MDAnalysis.Universe(psf, dcd)
    sel_h = u.select_atoms("(resname IM1H and name H7) or (resname IM1 and name H7) or (resname OAC and name H) or (resname HOAC and name H) or (resname HPTS and name H7) or (resname HPTSH and name H7) or (resname MEOH and name HO2) or (resname MEOH2 and name HO2)")
    sel_acid = u.select_atoms("(resname HPTSH)")
    n_acid =s el_acid.n_residues
    (...)
    ctr = 0
    for ts, step in zip(u.trajectory[::skip], list(charge_steps)[::skip]):
        
        indices = []
        for i in range(n_acid):
            indices.append(h_idx[ctr][i][0])

        sel_h_ts = sel_h[indices[0]]
        for i in range(1,n_acid):
            sel_h_ts = sel_h_ts + sel_h[indices[i]]

        pos_h[ctr] = sel_h_ts.positions

Then calculate the MSD using these coordinates.



