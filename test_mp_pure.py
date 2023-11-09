import numpy as np
import time
from pymatgen.core.lattice import Lattice
from pure_phase import XRDPattern

R = 100
p = 0
n = 0
times = []
for r in range(1, R + 1):
    print(f'round {r}, {n} invalid rounds, pass rate is {round(100*p/(r-n), 2)} %')
    start = time.time()
    a = np.random.uniform(2, 5)
    b = np.random.uniform(2, 5)
    c = np.random.uniform(2, 5)
    alpha = np.random.uniform(30, 100)
    beta = np.random.uniform(30, 100)
    gamma = np.random.uniform(30, 100)
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        rec_lattice = lattice.reciprocal_lattice
        qs = np.sort(rec_lattice.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 4*np.pi, zip_results = False)[1])[1::2]
    except:
        n += 1
        continue
    if len(qs) < 6:
        n += 1
        continue
    xrdp = XRDPattern(qs[:10])
    pred_rec_lattice = Lattice.from_parameters(*xrdp.get_rec_lattice())
    pred_lattice = pred_rec_lattice.reciprocal_lattice
    if lattice.find_mapping(pred_lattice):
        p += 1
    stop = time.time()
    times.append(stop - start)

print(f'After {R} trials, {n} invalid rounds, pass rate is {round(100*p/(R-n), 2)} %')
print(f'Average time: {np.mean(times)} +- {np.std(times)}')