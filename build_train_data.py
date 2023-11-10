import numpy as np
import torch
import json
from tqdm import tqdm
from pymatgen.core.structure import Structure
from data import Custom_Data
from mendeleev.fetch import fetch_table
from mendeleev import element
from sklearn.preprocessing import scale
from chemparse import parse_formula
import pickle

atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
atom_syms = {v: k for k, v in atom_nums.items()}
elem_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
                   'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat']
n_elem_feats = len(elem_feat_names) + 1
n_bond_feats = 32


target = "ebg"
DATAPATH = "./dataset/{}/".format(target)


def load_elem_feats(path_elem_embs=None):

    if path_elem_embs is None:
        # Get a table from Mendeleev features
        return get_mendeleev_feats()

    else:
        # Get a table of elemental features from elemental embeddings
        elem_feats = list()

        with open(path_elem_embs) as json_file:
            elem_embs = json.load(json_file)

            for elem in atom_nums.keys():
                elem_feats.append(np.array(elem_embs[elem]))

        return scale(np.vstack(elem_feats))


def get_mendeleev_feats():

    tb_atom_feats = fetch_table('elements')[:100]
    elem_feats = np.nan_to_num(np.array(tb_atom_feats[elem_feat_names]))
    ion_engs = np.zeros((elem_feats.shape[0], 1))

    for i in range(0, ion_engs.shape[0]):
        ion_eng = element(i + 1).ionenergies
        if 1 in ion_eng:
            ion_engs[i, 0] = element(i + 1).ionenergies[1]
        else:
            ion_engs[i, 0] = 0

    return scale(np.hstack([elem_feats, ion_engs]))


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):

        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

def load_dataset(mp_data, path_elem_embs="./res/matscholar-embedding.json"):
    
    elem_feats = load_elem_feats(path_elem_embs)
    dataset = list()

    error = 0

    for i in tqdm(range(0, len(mp_data))):
        
        mp_id = list(mp_data.keys())[i]
        str_cif = mp_data[mp_id]["cif"]
        formula = mp_data[mp_id]["pretty_formula"]

        try:            
            cg = get_crystal_graph(formula, elem_feats, str_cif, max_num_nbr=12, radius=8)
            cg.formula = formula
            cg.energy_per_atom = mp_data[mp_id]["energy_per_atom"]
            cg.band_gap = mp_data[mp_id]["band_gap"]
            dataset.append(cg)
        
        except:
            error += 1
            pass
    
    print("Converted data : {} || Total error : {}".format(len(dataset), error))

    return dataset


def get_crystal_graph(formula, elem_feats, str_cif, max_num_nbr, radius):

    gdf = GaussianDistance(dmin = 0.0, dmax = radius, step = 0.2)
    
    crystal = Structure.from_str(str_cif, fmt='cif')
    atom_fea = np.vstack([elem_feats[crystal[i].specie.number - 1]
                              for i in range(len(crystal))])
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                            [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                    nbr[:max_num_nbr])))
    
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    nbr_fea = gdf.expand(nbr_fea)

    atomic_nums = torch.tensor(crystal.atomic_numbers, dtype = torch.long)
    elem_batch = torch.zeros_like(atomic_nums)

    atom_feats = torch.Tensor(atom_fea)
    edge_attr = torch.Tensor(nbr_fea)
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
    index1 = torch.LongTensor([i for i in range(len(crystal))]).reshape(-1, 1).expand(nbr_fea_idx.shape).reshape(1, -1)
    index2 = torch.LongTensor(nbr_fea_idx).reshape(1, -1)
    edge_index = torch.cat([index1, index2], dim = 0)

    assert torch.isnan(atom_feats).sum() == 0
    assert torch.isnan(edge_attr).sum() == 0
    assert torch.isnan(edge_index).sum() == 0
    assert torch.isinf(atom_feats).sum() == 0
    assert torch.isinf(edge_attr).sum() == 0
    assert torch.isinf(edge_index).sum() == 0

    atomic_nums = torch.tensor(crystal.atomic_numbers, dtype = torch.long)

    fully_connected = []
    for i in range(len(atomic_nums.unique())):
        for j in range(len(atomic_nums.unique())):
            fully_connected.append(np.asarray([i, j]))
    
    parsed_formula = parse_formula(formula)

    sto_x = atomic_nums.unique()
    sto_batch = torch.zeros_like(sto_x)
    sto_bonds = torch.tensor(np.vstack(fully_connected).T)

    sto_weight = []
    for i in range(len(sto_x)):
        sto_weight.append(parsed_formula[list(atom_nums.keys())[list(atom_nums.values()).index(sto_x[i])]])
    sto_weight = torch.tensor(sto_weight, dtype=torch.float)

    return Custom_Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_attr.reshape(-1, edge_attr.shape[-1]), 
                       sto_x = sto_x, sto_edge_index = sto_bonds, sto_batch = sto_batch, sto_weight = sto_weight)


if __name__ == "__main__":
    
    with open("./dataset/mp.pkl", "rb") as f:
        mp_data = pickle.load(f)
    
    dataset = load_dataset(mp_data)
    torch.save(dataset, "./dataset/processed/pretrain_data_multi.pt")