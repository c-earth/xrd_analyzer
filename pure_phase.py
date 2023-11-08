import numpy as np

class XRDPattern():
    def __init__(self, qs, wl = 1):
        self.qs = qs
        self.qsv = qs[:10]
        self.wl = wl
        self.q_max = 4*np.pi/self.wl

    def gen_compatible_qs(self, tol = 1E-5):
        for i_c in range(len(self.qsv)):
            for i_b in range(0, i_c + 1):
                for i_a in range(0, i_b + 1):
                    q_a = self.qsv[i_a]
                    q_b = self.qsv[i_b]
                    q_c = self.qsv[i_c]
                    for q_d in self.qsv[((q_c-q_b+tol) < self.qsv) * (self.qsv < ((q_b**2+q_c**2)**0.5+tol))]:
                        for q_e in self.qsv[((q_c-q_a+tol) < self.qsv) * (self.qsv < ((q_c**2+q_a**2)**0.5+tol))]:
                            for q_f in self.qsv[((q_b-q_a+tol) < self.qsv) * (self.qsv < ((q_a**2+q_b**2)**0.5+tol))]:
                                if (q_d + q_e > q_f +tol) and (q_e + q_f > q_d +tol) and (q_f + q_d > q_e +tol):
                                    yield q_a, q_b, q_c, q_d, q_e, q_f

    def get_rec_lattice(self, tol = 1E-5):
        for i, (q_a, q_b, q_c, q_d, q_e, q_f) in enumerate(self.gen_compatible_qs(tol)):
            alpha = (q_b**2 + q_c**2 - q_d**2)
            beta = (q_c**2 + q_a**2 - q_e**2)
            gamma = (q_a**2 + q_b**2 - q_f**2)

            N_a = int(self.q_max // q_a)
            N_b = int(self.q_max // q_b)
            N_c = int(self.q_max // q_c)

            n_a = np.arange(-N_a, N_a + 1).reshape((-1, 1, 1))
            n_b = np.arange(-N_b, N_b + 1).reshape(( 1,-1, 1))
            n_c = np.arange(-N_c, N_c + 1).reshape(( 1, 1,-1))

            q_temp_sq = np.abs(n_a**2*q_a**2 + n_b**2*q_b**2 + n_c**2*q_c**2 + n_b*n_c*alpha + n_c*n_a*beta + n_a*n_b*gamma)
            q_temp = np.sqrt(q_temp_sq).reshape((1, -1))
            q_diff = np.min(np.abs(self.qs.reshape((-1, 1))-q_temp), axis = 1).flatten()
            if np.all(q_diff < tol):
                return q_a, q_b, q_c, np.arccos(alpha/(2*q_b*q_c))*180/np.pi, np.arccos(beta/(2*q_c*q_a))*180/np.pi, np.arccos(gamma/(2*q_a*q_b))*180/np.pi