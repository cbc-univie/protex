import os

import pytest

from protex.helpers import XMLSingleReader


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="No MDAnalysis",
)
def test_XMLSingleReader():
    a = XMLSingleReader("protex/forcefield/traj/im1h_oac_150_im1_hoac_350_npt_7.rst")

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="No MDAnalysis",
)
def test_Universe():
    import MDAnalysis
    u = MDAnalysis.Universe("protex/forcefield/im1h_oac_150_im1_hoac_350.psf","protex/forcefield/traj/im1h_oac_150_im1_hoac_350_npt_7.rst")
    #print(u.trajectory.n_frames)
    #print(u.trajectory.n_atoms)
    #print(dir(u.trajectory))
    sel = u.select_atoms("all")
    #print(sel.positions)