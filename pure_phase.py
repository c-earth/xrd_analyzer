import numpy as np

class XRDPattern():
    def __init__(self, qs, wl = 1):
        self.qs = qs
        self.wl = wl
        self.q_max = 4*np.pi/self.wl

    def gen_compatible_qs(self):
        for i_a in range(len(self.qs)):
            for i_b in range(i_a, len(self.qs)):
                for q_c in self.qs[i_b:]:
                    q_a = self.qs[i_a]
                    q_b = self.qs[i_b]
                    for q_d in self.qs[(q_a <= self.qs) * (self.qs < (q_b+q_c))]:
                        for q_e in self.qs[(q_a <= self.qs) * (self.qs < (q_c+q_a))]:
                            for q_f in self.qs[(q_a <= self.qs) * (self.qs < (q_a+q_b))]:
                                yield q_a, q_b, q_c, q_d, q_e, q_f

    def get_lattice(self, tol = 1E-5):
        for i, (q_a, q_b, q_c, q_d, q_e, q_f) in enumerate(self.gen_compatible_qs()):
            alpha = (q_b**2 + q_c**2 - q_d**2)
            beta = (q_c**2 + q_a**2 - q_e**2)
            gamma = (q_a**2 + q_b**2 - q_f**2)

            N_a = int(self.q_max // q_a)
            N_b = int(self.q_max // q_b)
            N_c = int(self.q_max // q_c)

            n_a = np.arange(-N_a, N_a + 1).reshape((-1, 1, 1))
            n_b = np.arange(-N_b, N_b + 1).reshape(( 1,-1, 1))
            n_c = np.arange(-N_c, N_c + 1).reshape(( 1, 1,-1))

            q_temp_sq = n_a**2*q_a**2 + n_b**2*q_b**2 + n_c**2*q_c**2 + n_b*n_c*alpha + n_c*n_a*beta + n_a*n_b*gamma
            q_temp = np.sqrt(q_temp_sq).reshape((1, -1))
            q_diff = np.min(np.abs(self.qs.reshape((-1, 1))-q_temp), axis = 1).flatten()
            if np.all(q_diff < tol):
                print(f'end at round {i+1}')
                return q_a, q_b, q_c, np.arccos(alpha/(2*q_b*q_c))*180/np.pi, np.arccos(beta/(2*q_c*q_a))*180/np.pi, np.arccos(gamma/(2*q_a*q_b))*180/np.pi