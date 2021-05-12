# Import package, test suite, and other packages as needed
import protex
import pytest
import sys
from sys import stdout


def load_charmm_files():
    from simtk.openmm.app import CharmmParameterSet, CharmmPsfFile, CharmmCrdFile

    # =======================================================================
    # Force field
    # =======================================================================

    # Loading CHARMM files
    print("Loading CHARMM files...")
    PARA_FILES = [
        "toppar_drude_master_protein_2013f_lj02.str",
        "hoac_d.str",
        "im1h_d_fm_lj.str",
        "im1_d_fm_lj.str",
        "oac_d_lj.str",
    ]
    base = "protex/charmm_ff"
    params = CharmmParameterSet(
        *[f"{base}/toppar/{para_files}" for para_files in PARA_FILES]
    )

    psf = CharmmPsfFile(f"{base}/im1h_oac_150_im1_hoac_350.psf")
    # cooridnates can be provieded by CharmmCrdFile, CharmmRstFile or PDBFile classes
    crd = CharmmCrdFile(f"{base}/im1h_oac_150_im1_hoac_350.crd")
    return psf, crd, params


def setup_system():
    from simtk.unit import angstroms
    from simtk.openmm.app import PME, HBonds

    psf, crd, params = load_charmm_files()
    xtl = 48.0 * angstroms
    psf.setBox(xtl, xtl, xtl)
    system = psf.createSystem(
        params,
        nonbondedMethod=PME,
        nonbondedCutoff=11.0 * angstroms,
        switchDistance=10 * angstroms,
        constraints=HBonds,
    )

    return system


def setup_simulation():
    from simtk.unit import kelvin, picoseconds
    from simtk.openmm.app import Simulation
    from simtk.openmm import DrudeLangevinIntegrator

    psf, crd, params = load_charmm_files()
    system = setup_system()
    integrator = DrudeLangevinIntegrator(
        300 * kelvin,
        1 / picoseconds,
        300 * kelvin,
        1 / picoseconds,
        0.002 * picoseconds,
    )
    simulation = Simulation(psf.topology, system, integrator)
    simulation.context.setPositions(crd.positions)
    return simulation


def test_setup_simulation():

    simulation = setup_simulation()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)


def test_run_simulation():
    from simtk.openmm.app import StateDataReporter, PDBReporter, DCDReporter

    simulation = setup_simulation()
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=100)
    simulation.reporters.append(PDBReporter("output.pdb", 1))
    simulation.reporters.append(DCDReporter("output.dcd", 1))

    simulation.reporters.append(
        StateDataReporter(
            stdout,
            1,
            step=True,
            potentialEnergy=True,
            temperature=True,
            time=True,
            volume=True,
            density=True,
        )
    )
    print("Running dynmamics...")
    simulation.step(10)
