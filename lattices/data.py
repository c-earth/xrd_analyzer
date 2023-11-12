import numpy as np

from pymatgen.core.lattice import Lattice

class LatticeGen():
    def __init__(self, q_abc_res, q_ang_res):
        self.q_abc_res = q_abc_res
        self.q_ang_res = q_ang_res
        
    def gen_rec_lattice_params(self):
        discrete_q_abc = np.random.randint(1, self.q_abc_res + 1, size = (3))
        discrete_q_ang = np.random.randint(1, self.q_ang_res + 1, size = (3))
        return discrete_q_abc, discrete_q_ang

    def gen_data(self, discrete_q_abc, discrete_q_ang):
        rec_lattice = Lattice.from_parameters(*discrete_q_abc/100, *discrete_q_ang)
        raw_qs = np.unique(np.round(rec_lattice.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 2, zip_results = False)[1], decimals = 5))
        y = np.zeros((self.q_abc_res, self.q_ang_res))
        y[discrete_q_abc-1, discrete_q_ang-1] = 1/3
        return raw_qs, y

    def gen_valid_random_data(self):
        while True:
            try:
                discrete_q_abc, discrete_q_ang = self.gen_rec_lattice_params()
                x, y = self.gen_data(discrete_q_abc, discrete_q_ang)
                if len(x) == 0:
                    continue
                return x, y
            except:
                continue