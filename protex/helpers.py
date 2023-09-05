"""helpers.py contains additional functions for easier use of protex."""
from MDAnalysis.coordinates.base import SingleFrameReaderBase
from MDAnalysis.lib.util import openany
from openmm import XmlSerializer
from parmed.formats import PSFFile
from parmed.topologyobjects import DrudeAtom, ExtraPoint, LocalCoordinatesFrame

# from parmed.charmm.psf import CharmmPsfFile
# from parmed.formats.registry import FileFormatType
from parmed.utils import tag_molecules
from parmed.utils.io import genopen


class XMLSingleReader(SingleFrameReaderBase):
    format = "rst"
    units = {"time": "ps", "length": "Nanometer"}
    _format_type = "xml"

    def _read_first_frame(self):
        with openany(self.filename) as xmlfile:
            state = XmlSerializer.deserialize(xmlfile.read())
            positions = state.getPositions(asNumpy=True)
        self.n_atoms = positions.shape[0]
        self.ts = self._Timestep.from_coordinates(positions, **self._ts_kwargs)
        self.ts.frame = 0


class CustomPSFFile(PSFFile):
    """
    Child of parmeds formats/psf.py PFSFile class
    used to get psf files if NUMANSIO section is empty
    -> remove if parmed includes https://github.com/ParmEd/ParmEd/pull/1274.
    """

    @staticmethod
    def write(struct, dest, vmd=False):
        """
        Writes a PSF file from the stored molecule.

        Parameters
        ----------
        struct : :class:`Structure`
            The Structure instance from which the PSF should be written
        dest : str or file-like
            The place to write the output PSF file.  If it has a "write"
            attribute, it will be used to print the PSF file. Otherwise, it will
            be treated like a string and a file will be opened, printed, then
            closed
        vmd : bool
            If True, it will write out a PSF in the format that VMD prints it in
            (i.e., no NUMLP/NUMLPH or MOLNT sections).

        Examples
        --------
        >>> cs = CharmmPsfFile('testfiles/test.psf')
        >>> cs.write_psf('testfiles/test2.psf').
        """
        # See if this is an extended format
        try:
            ext = "EXT" in struct.flags
        except AttributeError:
            ext = True
        # See if this is an XPLOR format
        try:
            xplor = "XPLOR" in struct.flags
        except AttributeError:
            for atom in struct.atoms:
                if isinstance(atom.type, str):
                    xplor = True
                    break
            else:
                xplor = False
        own_handle = False
        # Index the atoms and residues TODO delete
        if isinstance(dest, str):
            own_handle = True
            dest = genopen(dest, "w")

        # Assign the formats we need to write with
        if ext:
            atmfmt1 = "%10d %-8s %-8i %-8s %-8s %6d %10.6f %13.4f" + 11 * " "
            atmfmt2 = "%10d %-8s %-8i %-8s %-8s %-6s %10.6f %13.4f" + 11 * " "
            intfmt = "%10d"  # For pointers
        else:
            atmfmt1 = "%8d %-4s %-4i %-4s %-4s %4d %10.6f %13.4f" + 11 * " "
            atmfmt2 = "%8d %-4s %-4i %-4s %-4s %-4s %10.6f %13.4f" + 11 * " "
            intfmt = "%8d"  # For pointers

        # Now print the header then the title
        dest.write("PSF CHEQ ")
        if hasattr(struct, "flags"):
            dest.write(" ".join(f for f in struct.flags if f not in ("CHEQ",)))
        else:
            dest.write("EXT")  # EXT is always active if no flags present
            if xplor:
                dest.write(" XPLOR")
        dest.write("\n\n")
        if isinstance(struct.title, str):
            dest.write(intfmt % 1 + " !NTITLE\n")
            dest.write("%s\n\n" % struct.title)
        else:
            dest.write(intfmt % len(struct.title) + " !NTITLE\n")
            dest.write("\n".join(struct.title) + "\n\n")
        # Now time for the atoms
        dest.write(intfmt % len(struct.atoms) + " !NATOM\n")
        # atmfmt1 is for CHARMM format (i.e., atom types are integers)
        # atmfmt is for XPLOR format (i.e., atom types are strings)
        add = 0 if struct.residues[0].number > 0 else 1 - struct.residues[0].number
        for i, atom in enumerate(struct.atoms):
            typ = atom.type
            if isinstance(atom.type, str):
                fmt = atmfmt2
                if not atom.type:
                    typ = atom.name
            else:
                fmt = atmfmt1
            segid = atom.residue.segid or "SYS"
            atmstr = fmt % (
                i + 1,
                segid,
                atom.residue.number + add,
                atom.residue.name,
                atom.name,
                typ,
                atom.charge,
                atom.mass,
            )
            if hasattr(atom, "props"):
                dest.write(atmstr + "   ".join(atom.props) + "\n")
            else:
                dest.write("%s\n" % atmstr)
        dest.write("\n")
        # Bonds
        dest.write(intfmt % len(struct.bonds) + " !NBOND: bonds\n")
        for i, bond in enumerate(struct.bonds):
            dest.write((intfmt * 2) % (bond.atom1.idx + 1, bond.atom2.idx + 1))
            if i % 4 == 3:  # Write 4 bonds per line
                dest.write("\n")
        # See if we need to terminate
        if len(struct.bonds) % 4 != 0 or len(struct.bonds) == 0:
            dest.write("\n")
        dest.write("\n")
        # Angles
        dest.write(intfmt % len(struct.angles) + " !NTHETA: angles\n")
        for i, angle in enumerate(struct.angles):
            dest.write(
                (intfmt * 3)
                % (angle.atom1.idx + 1, angle.atom2.idx + 1, angle.atom3.idx + 1)
            )
            if i % 3 == 2:  # Write 3 angles per line
                dest.write("\n")
        # See if we need to terminate
        if len(struct.angles) % 3 != 0 or len(struct.angles) == 0:
            dest.write("\n")
        dest.write("\n")
        # Dihedrals
        # impropers need to be split off in the "improper" section.
        # PSF files need to have each dihedral listed *only* once. So count the
        # number of unique dihedrals
        nnormal = 0
        torsions = set()
        for dih in struct.dihedrals:
            if dih.improper:
                continue
            a1, a2, a3, a4 = dih.atom1, dih.atom2, dih.atom3, dih.atom4
            if (a1, a2, a3, a4) in torsions or (a4, a3, a2, a1) in torsions:
                continue
            nnormal += 1
            torsions.add((a1, a2, a3, a4))
        nimprop = sum(1 for dih in struct.dihedrals if dih.improper)
        dest.write(intfmt % nnormal + " !NPHI: dihedrals\n")
        torsions = set()
        c = 0
        for dih in struct.dihedrals:
            if dih.improper:
                continue
            a1, a2, a3, a4 = dih.atom1, dih.atom2, dih.atom3, dih.atom4
            if (a1, a2, a3, a4) in torsions or (a4, a3, a2, a1) in torsions:
                continue
            dest.write((intfmt * 4) % (a1.idx + 1, a2.idx + 1, a3.idx + 1, a4.idx + 1))
            torsions.add((a1, a2, a3, a4))
            if c % 2 == 1:  # Write 2 dihedrals per line
                dest.write("\n")
            c += 1
        # See if we need to terminate
        if nnormal % 2 != 0 or nnormal == 0:
            dest.write("\n")
        dest.write("\n")
        # Impropers
        nimprop += len(struct.impropers)
        dest.write(intfmt % (nimprop) + " !NIMPHI: impropers\n")

        def improp_gen(struct):
            for imp in struct.impropers:
                yield (imp.atom1, imp.atom2, imp.atom3, imp.atom4)
            for dih in struct.dihedrals:
                if dih.improper:
                    yield (dih.atom1, dih.atom2, dih.atom3, dih.atom4)

        for i, (a1, a2, a3, a4) in enumerate(improp_gen(struct)):
            dest.write((intfmt * 4) % (a1.idx + 1, a2.idx + 1, a3.idx + 1, a4.idx + 1))
            if i % 2 == 1:  # Write 2 dihedrals per line
                dest.write("\n")
        # See if we need to terminate
        if nimprop % 2 != 0 or nimprop == 0:
            dest.write("\n")
        dest.write("\n")
        # Donor section
        dest.write(intfmt % len(struct.donors) + " !NDON: donors\n")
        for i, don in enumerate(struct.donors):
            dest.write((intfmt * 2) % (don.atom1.idx + 1, don.atom2.idx + 1))
            if i % 4 == 3:  # 4 donors per line
                dest.write("\n")
        if len(struct.donors) % 4 != 0 or len(struct.donors) == 0:
            dest.write("\n")
        dest.write("\n")
        # Acceptor section
        dest.write(intfmt % len(struct.acceptors) + " !NACC: acceptors\n")
        for i, acc in enumerate(struct.acceptors):
            dest.write((intfmt * 2) % (acc.atom1.idx + 1, acc.atom2.idx + 1))
            if i % 4 == 3:  # 4 donors per line
                dest.write("\n")
        if len(struct.acceptors) % 4 != 0 or len(struct.acceptors) == 0:
            dest.write("\n")
        dest.write("\n")
        # NNB section ??
        dest.write(intfmt % 0 + " !NNB\n\n")
        for i in range(len(struct.atoms)):
            dest.write(intfmt % 0)
            if i % 8 == 7:  # Write 8 0's per line
                dest.write("\n")
        if len(struct.atoms) % 8 != 0:
            dest.write("\n")
        dest.write("\n")
        # Group section
        try:
            nst2 = struct.groups.nst2
        except AttributeError:
            nst2 = 0
        dest.write((intfmt * 2) % (len(struct.groups) or 1, nst2))
        dest.write(" !NGRP NST2\n")
        if struct.groups:
            for i, gp in enumerate(struct.groups):
                dest.write((intfmt * 3) % (gp.atom.idx, gp.type, gp.move))
                if i % 3 == 2:
                    dest.write("\n")
            if len(struct.groups) % 3 != 0 or len(struct.groups) == 0:
                dest.write("\n")
        else:
            typ = 1 if abs(sum(a.charge for a in struct.atoms)) < 1e-4 else 2
            dest.write((intfmt * 3) % (0, typ, 0))
            dest.write("\n")
        dest.write("\n")
        # The next two sections are never found in VMD prmtops...
        if not vmd:
            # Molecule section; first set molecularity
            tag_molecules(struct)
            mollist = [a.marked for a in struct.atoms]
            dest.write(intfmt % max(mollist) + " !MOLNT\n")
            for i, atom in enumerate(struct.atoms):
                dest.write(intfmt % atom.marked)
                if i % 8 == 7:
                    dest.write("\n")
            if len(struct.atoms) % 8 != 0:
                dest.write("\n")
            dest.write("\n")
            # NUMLP/NUMLPH section
            lone_pairs = [atom for atom in struct.atoms if isinstance(atom, ExtraPoint)]

            def get_host_atoms(frame_type):
                if isinstance(frame_type, LocalCoordinatesFrame):
                    return frame_type.get_atoms()[: frame_type.frame_size]
                return frame_type.get_atoms()

            frame_hosts = [
                [fatom for fatom in get_host_atoms(atom.frame_type)]
                for atom in lone_pairs
            ]
            dest.write(
                (intfmt * 2) % (len(lone_pairs), sum(len(x) + 1 for x in frame_hosts))
                + " !NUMLP NUMLPH\n"
            )
            i = 1
            if lone_pairs:
                lph_section = []
                for lone_pair, frame_host in zip(lone_pairs, frame_hosts):
                    if not isinstance(lone_pair.frame_type, LocalCoordinatesFrame):
                        raise TypeError(
                            "CHARMM PSF files only support LocalCoordinatesFrame LPs"
                        )
                    dest.write((intfmt * 2) % (len(frame_host), i))
                    dest.write(
                        "%4s%10.5f%14.5f%14.5f\n"
                        % (
                            "F",
                            lone_pair.frame_type.distance,
                            lone_pair.frame_type.angle,
                            lone_pair.frame_type.dihedral,
                        )
                    )
                    frame = [lone_pair] + frame_host[: lone_pair.frame_type.frame_size]
                    if len(frame) == 4:
                        frame = [frame[0], frame[1], frame[3], frame[2]]
                    for atom in frame:
                        lph_section.append(intfmt % (atom.idx + 1))
                        if i % 8 == 0:
                            lph_section.append("\n")
                        i += 1
                if lph_section[-1] == "\n":
                    lph_section.pop()
                lph_section.append("\n")
                dest.write("".join(lph_section))
            dest.write("\n")
            drudes = [atom for atom in struct.atoms if isinstance(atom, DrudeAtom)]
            if drudes:
                n_aniso = sum(drude.anisotropy is not None for drude in drudes)
                dest.write((intfmt % n_aniso) + " !NUMANISO\n")
                # FIX
                if (
                    n_aniso
                ):  # this check is new, see https://github.com/ParmEd/ParmEd/pull/1274
                    idx_section = []
                    i = 1
                    for drude in drudes:
                        if drude.anisotropy is None:
                            continue
                        if any(
                            key not in drude.anisotropy.params
                            for key in ["k11", "k22", "k33"]
                        ):
                            raise ValueError(
                                "CHARMM Drude anisotropy parameters cannot be determined from just "
                                "two polarizability scale factors."
                            )
                        dest.write(" " * 8)
                        dest.write(
                            "%14.5f%14.5f%14.5f\n"
                            % (
                                drude.anisotropy.params["k11"],
                                drude.anisotropy.params["k22"],
                                drude.anisotropy.params["k33"],
                            )
                        )
                        for atom in [
                            drude.anisotropy.atom1,
                            drude.anisotropy.atom2,
                            drude.anisotropy.atom3,
                            drude.anisotropy.atom4,
                        ]:
                            idx_section.append(intfmt % (atom.idx + 1))
                            if i % 8 == 0:
                                idx_section.append("\n")
                            i += 1
                    if idx_section[-1] == "\n":
                        idx_section.pop()
                    idx_section.append("\n")
                    dest.write("".join(idx_section))
                dest.write("\n")
        # CMAP section
        dest.write(intfmt % len(struct.cmaps) + " !NCRTERM: cross-terms\n")
        for i, cmap in enumerate(struct.cmaps):
            dest.write(
                (intfmt * 8)
                % (
                    cmap.atom1.idx + 1,
                    cmap.atom2.idx + 1,
                    cmap.atom3.idx + 1,
                    cmap.atom4.idx + 1,
                    cmap.atom2.idx + 1,
                    cmap.atom3.idx + 1,
                    cmap.atom4.idx + 1,
                    cmap.atom5.idx + 1,
                )
            )
            dest.write("\n")
        dest.write("\n")
        # Done!
        # If we opened our own handle, close it
        if own_handle:
            dest.close()
