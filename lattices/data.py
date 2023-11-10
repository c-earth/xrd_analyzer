import numpy as np

from pymatgen.core.lattice import Lattice

def gen_rec_lattice_params():
    while True:
        discrete_q_abc = np.random.randint(1, 101, size = (3))
        discrete_q_ang = np.random.randint(1, 91, size = (3))
        sort_q_angs = np.sort(discrete_q_ang)
        if sort_q_angs[0] + sort_q_angs[1] < sort_q_angs[2]+5:
            continue
        else:
            return discrete_q_abc, discrete_q_ang

def gen_data(discrete_q_abcs, discrete_q_angs):
    X = []
    Y = []
    for discrete_q_abc, discrete_q_ang, in zip(discrete_q_abcs, discrete_q_angs):
        rec_lattice = Lattice.from_parameters(*discrete_q_abc/100, *discrete_q_ang)
        raw_qs = np.unique(np.round(rec_lattice.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 2, zip_results = False)[1], decimals = 5))
        if len(raw_qs) < 40:
            raw_qs = np.concatenate([np.zeros((40 - len(raw_qs))), raw_qs])
        x = raw_qs[:40].reshape((1, -1))
        y = np.zeros((1, 100, 90))
        y[[0, 0, 0], discrete_q_abc-1, discrete_q_ang-1] = 1
        X.append(x)
        Y.append(y)
    return np.vstack(X), np.vstack(Y)

def gen_batch(batch_size = 1):
    discrete_q_abcs = []
    discrete_q_angs = []
    for _ in range(batch_size):
        discrete_q_abc, discrete_q_ang = gen_rec_lattice_params()
        discrete_q_abcs.append(discrete_q_abc)
        discrete_q_angs.append(discrete_q_ang)
    return gen_data(discrete_q_abcs, discrete_q_angs)