import numpy as np
from pymatgen.core.lattice import Lattice

class XRDPattern():
    def __init__(self, qs, wl = 1):
        self.qs = qs
        self.wl = wl
        self.q_max = 4*np.pi/self.wl

    def gen_compatible_qs(self, tol = 1E-3):
        for i_c in range(len(self.qs)):
            for i_b in range(0, i_c + 1):
                for i_a in range(0, i_b + 1):
                    q_a = self.qs[i_a]
                    q_b = self.qs[i_b]
                    q_c = self.qs[i_c]
                    for q_d in self.qs[((q_c-q_b+tol) < self.qs) * (self.qs < ((q_c**2+q_b**2)**0.5+tol))]:
                        for q_e in self.qs[((q_c-q_a+tol) < self.qs) * (self.qs < ((q_c**2+q_a**2)**0.5+tol))]:
                            x_1 = (q_a**2+q_c**2-q_e**2)/(2*q_c)
                            x_2 = (q_b**2+q_c**2-q_d**2)/(2*q_c)
                            y_1 = (abs(q_a**2-x_1**2))**0.5
                            y_2 = (abs(q_b**2-x_2**2))**0.5
                            lower = (abs(q_a**2+q_b**2-2*x_1*x_2-2*y_1*y_2))**0.5
                            upper = (abs(q_a**2+q_b**2-2*x_1*x_2))**0.5
                            for q_f in self.qs[(lower + tol < self.qs) * (self.qs < upper + tol)]:
                                yield q_a, q_b, q_c, q_d, q_e, q_f

    def get_rec_lattice(self, tol = 1E-3):
        for q_a, q_b, q_c, q_d, q_e, q_f in self.gen_compatible_qs():
            cq_alpha = max(min((q_b**2 + q_c**2 - q_d**2)/(2*q_b*q_c), 1), -1)
            cq_beta = max(min((q_c**2 + q_a**2 - q_e**2)/(2*q_c*q_a), 1), -1)
            cq_gamma = max(min((q_a**2 + q_b**2 - q_f**2)/(2*q_a*q_b), 1), -1)

            q_alpha = np.arccos(cq_alpha)*180/np.pi
            q_beta = np.arccos(cq_beta)*180/np.pi
            q_gamma = np.arccos(cq_gamma)*180/np.pi

            rec_lattice_tmp = Lattice.from_parameters(q_a, q_b, q_c, q_alpha, q_beta, q_gamma)
            points = rec_lattice_tmp.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 4*np.pi, zip_results = False)
            q_tmp = points[1].reshape((1, -1))

            q_dif_all = np.abs(self.qs.reshape((-1, 1))-q_tmp)
            q_dif_idx = np.argmin(q_dif_all, axis = 1).flatten()
            q_dif = np.array([q_dif_all[i, min_idx_i] for i, min_idx_i in enumerate(q_dif_idx)])
            if np.all(q_dif < tol):
                ns = points[0][q_dif_idx]
                converter = np.zeros((6, 3, 3))
                converter[0, 0, 0] = 1
                converter[1, 1, 1] = 1
                converter[2, 2, 2] = 1
                converter[3, 1, 2] = 2
                converter[4, 2, 0] = 2
                converter[5, 0, 2] = 2
                X = np.einsum('ijk,lj,mk->ilm', converter, q_dif_idx, q_dif_idx)