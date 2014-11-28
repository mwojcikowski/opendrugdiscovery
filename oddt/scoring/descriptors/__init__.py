import numpy as np
from scipy.spatial.distance import cdist as distance

def atoms_by_type(atom_dict, types, mode = 'atomic_nums'):
    """
    AutoDock4 types definition: http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters
    """
    if mode == 'atomic_nums':
        return {num: atom_dict[atom_dict['atomicnum'] == num] for num in set(types)}
    elif mode == 'atom_types_sybyl':
        return {t: atom_dict[atom_dict['atomtype'] == t] for t in set(types)}
    elif mode == 'atom_types_ad4':
        # all AD4 atom types are capitalized
        types = [t.upper() for t in types]
        out = {}
        for t in set(types):
            if t == 'HD':
                out[t] = atom_dict[atom_dict['atomicnum'] == 1 & atom_dict['isdonorh']]
            elif t == 'C':
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & ~atom_dict['isaromatic']]
            elif t == 'CD': # not canonical AD4 type, although used by NNscore, with no description. properies assued by name
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & ~atom_dict['isdonor']]
            elif t == 'A':
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & atom_dict['isaromatic']]
            elif t == 'N':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & ~atom_dict['isacceptor']]
            elif t == 'NA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & atom_dict['isacceptor']]
            elif t == 'OA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 8 & atom_dict['isacceptor']]
            elif t == 'F':
                out[t] = atom_dict[atom_dict['atomicnum'] == 9]
            elif t == 'MG':
                out[t] = atom_dict[atom_dict['atomicnum'] == 12]
            elif t == 'P':
                out[t] = atom_dict[atom_dict['atomicnum'] == 15]
            elif t == 'SA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 16 & atom_dict['isacceptor']]
            elif t == 'S':
                out[t] = atom_dict[atom_dict['atomicnum'] == 16 & ~atom_dict['isacceptor']]
            elif t == 'CL':
                out[t] = atom_dict[atom_dict['atomicnum'] == 17]
            elif t == 'CA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 20]
            elif t == 'MN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 25]
            elif t == 'FE':
                out[t] = atom_dict[atom_dict['atomicnum'] == 26]
            elif t == 'CU':
                out[t] = atom_dict[atom_dict['atomicnum'] == 29]
            elif t == 'ZN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 30]
            elif t == 'BR':
                out[t] = atom_dict[atom_dict['atomicnum'] == 35]
            elif t == 'I':
                out[t] = atom_dict[atom_dict['atomicnum'] == 53]
            else:
                 raise ValueError('Unsopported atom type: %s' % t)
        return out

class close_contacts(object):
    def __init__(self, protein = None, cutoff = 4, mode = 'atomic_nums', ligand_types = None, protein_types = None, aligned_pairs = False):
        self.cutoff = cutoff
        self.protein = protein
        self.ligand_types = ligand_types
        self.protein_types = protein_types if protein_types else ligand_types
        self.aligned_pairs = aligned_pairs
        self.mode = mode
    
    def build(self, ligands, protein = None, single = False):
        if protein is None:
            protein = self.protein
        if single:
            ligands = [ligands]
#        prot_dict = atoms_by_type(protein.atom_dict, self.protein_types, self.mode)
        desc_size = len(self.ligand_types) if self.aligned_pairs else len(self.ligand_types)*len(self.protein_types)
        out = np.zeros(desc_size, dtype=int)
        for mol in ligands:
#            mol_dict = atoms_by_type(mol.atom_dict, self.ligand_types, self.mode) 
            if self.aligned_pairs:
                #desc = np.array([(distance(prot_dict[str(prot_type)]['coords'], mol_dict[str(mol_type)]['coords']) <= self.cutoff).sum() for mol_type, prot_type in zip(self.ligand_types, self.protein_types)], dtype=int)
                # this must be LAZY!
                desc = np.array([(distance(atoms_by_type(protein.atom_dict, [prot_type], self.mode)[prot_type]['coords'], atoms_by_type(mol.atom_dict, [mol_type], self.mode)[mol_type]['coords']) <= self.cutoff).sum() for mol_type, prot_type in zip(self.ligand_types, self.protein_types)], dtype=int)
            else:
                desc = np.array([(distance(atoms_by_type(protein.atom_dict, [prot_type], self.mode)[prot_type]['coords'], atoms_by_type(mol.atom_dict, [mol_type], self.mode)[mol_type]['coords']) <= self.cutoff).sum() for mol_type in self.ligand_types for prot_type in self.protein_types], dtype=int)
            out = np.vstack((out, desc))
        return out[1:]
    
    def __reduce__(self):
        return close_contacts, (None, self.cutoff, self.mode, self.ligand_types, self.protein_types, self.aligned_pairs)
        
class fingerprints(object):
    def __init__(self, fp = 'fp2', toolkit = 'ob'):
        self.fp = fp
        self.exchange = False
        #if toolkit == oddt.toolkit.backend:
        #    self.exchange = False
        #else:
        #    self.exchange = True
        #    self.target_toolkit = __import__('toolkits.'+toolkit)
    
    def _get_fingerprint(self, mol):
        if self.exchange:
            mol = self.target_toolkit.Molecule(mol)
        return mol.calcfp(self.fp).raw
        
    def build(self, mols, single = False):
        if single:
            mols = [mols]
        out = None
        
        for mol in mols:
            fp = self._get_fingerprint(mol)
            if out is None:
                out = np.zeros_like(fp)
            out = np.vstack((fp, out))
        return out[1:]
    
    def __reduce__(self):
        return fingerprints, ()
