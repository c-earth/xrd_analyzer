import pickle as pkl

from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice

from pure_phase import XRDPattern

with open('./data/mpid_structure.pkl', 'rb') as f:
    mpid_structure_dict = pkl.load(f)

xrd_calc = XRDCalculator(wavelength = 1, symprec = 1E-5)

def gen_xrd_pattern(structure):
    sga = SpacegroupAnalyzer(structure, symprec = 1E-5)
    primitive_structure = sga.find_primitive()
    pattern = xrd_calc.get_pattern(primitive_structure, scaled = False, two_theta_range = None)
    return pattern.x

for mpid, structure in mpid_structure_dict.items():
    twothetas = gen_xrd_pattern(structure)
    xrdp = XRDPattern.from_twothetas(twothetas, wl = 1)
    q_a, q_b, q_c, q_alpha, q_beta, q_gamma = xrdp.get_rec_lattice()
    rec_lattice = Lattice.from_parameters(q_a, q_b, q_c, q_alpha, q_beta, q_gamma)
    lattice = rec_lattice.reciprocal_lattice
    print(structure.lattice)
    print(lattice)
    break