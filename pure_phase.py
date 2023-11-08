import numpy as np

class XRDPattern():
    def __init__(self, qs):
        self.qs = qs

    def gen_compatible_qs(self):
        for i_a in range(len(self.qs)):
            for i_b in range(i_a, len(self.qs)):
                for q_c in self.qs[i_b:]:
                    q_a = self.qs[i_a]
                    q_b = self.qs[i_b]
                    for q_d in self.qs[q_a <= self.qs < (q_b+q_c)]:
                        for q_e in self.qs[q_a <= self.qs < (q_c+q_a)]:
                            for q_f in self.qs[q_a <= self.qs < (q_a+q_b)]:
                                yield q_a, q_b, q_c, q_d, q_e, q_f

    def get_lattice(self):
        for q_a, q_b, q_c, q_d, q_e, q_f in self.gen_compatible_qs():
            q_alpha = np.arccos((q_b**2 + q_c**2 - q_d**2)/(2*q_b*q_c))
            q_beta = np.arccos((q_c**2 + q_a**2 - q_e**2)/(2*q_c*q_a))
            q_gamma = np.arccos((q_a**2 + q_b**2 - q_f**2)/(2*q_a*q_b))

            