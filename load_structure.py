from mp_api.client import MPRester
import pickle as pkl

with MPRester('KKYDb56RGe2YQhuLJqE17k7AztkJ6Fyj') as mpr:
    docs = mpr.summary.search(fields=['material_id', 'structure'])
    mpid_structure_dict = {doc.material_id: doc.structure for doc in docs}

with open('data/mpid_structure.pkl', 'wb') as f:
    pkl.dump(mpid_structure_dict, f)