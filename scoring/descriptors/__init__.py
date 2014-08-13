import numpy as np
from scipy.spatial.distance import cdist as distance

class close_contacts:
    def __init__(self, protein = None, cutoff = 4, mode = 'atomic_nums', ligand_types = None, protein_types = None, aligned_pairs = False):
        self.cutoff = cutoff
        self.protein = protein
        self.ligand_types = ligand_types
        self.protein_types = protein_types if protein_types else ligand_types
        self.aligned_pairs = aligned_pairs
        
        if mode == 'atomic_nums':
            self.typer = self._atomic_nums
        elif mode == 'atom_types_sybyl':
            self.typer = self._atom_types_sybyl
        elif mode == 'atom_types_ad4':
            self.typer = self._atom_types_ad4
    
    def _atomic_nums(atom_dict, protein = False):
        atomic_nums = self.protein_types if protein else self.ligand_types
        return {num: atom_dict[atom_dict['atomicnum'] == num] for num in atomic_nums}
    
    def _atomic_nums(self, atom_dict, protein = False):
        atomic_nums = self.protein_types if protein else self.ligand_types
        return {num: atom_dict[atom_dict['atomicnum'] == num] for num in atomic_nums}
    
    def _atom_types_sybyl(self, atom_dict, protein = False):
        atom_types = self.protein_types if protein else self.ligand_types
        return {t: atom_dict[atom_dict['atomtype'] == t] for t in atom_types}
    
    def _atom_types_ad4(self, atom_dict, protein = False):
        """
        AutoDock4 types definition: http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters
        """
        atom_types = self.protein_types if protein else self.ligand_types
        # all AD4 atom types are capitalized
        atom_types = [i.upper() for i in atom_types]
        out = {}
        for t in atom_types:
            if t == 'HD':
                out[t] = atom_dict[atom_dict['atomicnum'] == 1 & atom_dict['isdonorh']]
            elif t == 'C':
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & ~atom_dict['isaromatic']]
            elif t == 'A':
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & atom_dict['isaromatic']]
            elif t == 'N':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & ~atom_dict['isacceptor']]
            elif t == 'NA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & atom_dict['isacceptor']]
            elif t == 'NA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & atom_dict['isacceptor']]
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
            elif t == 'ZN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 30]
            elif t == 'BR':
                out[t] = atom_dict[atom_dict['atomicnum'] == 35]
            elif t == 'I':
                out[t] = atom_dict[atom_dict['atomicnum'] == 53]
        return out
    
    def build(self, ligands, protein = None):
        if protein is None:
            protein = self.protein
        prot_dict = self.typer(protein.atom_dict, protein = True)
        out = np.zeros(len(self.ligand_types)*len(self.protein_types), dtype=int)
        for mol in ligands:
            mol_dict = self.typer(mol.atom_dict) 
            desc = np.array([(distance(prot_dict[prot_type]['coords'], mol_dict[mol_type]['coords']) <= self.cutoff).sum() for prot_type in self.protein_types for mol_type in self.ligand_types], dtype=int)
            out = np.vstack((out, desc))
        return out[1:]
        
class fingerprints:
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
        
    def build(self, mols):
        out = None
        for mol in mols:
            fp = self._get_fingerprint(mol)
            if out is None:
                out = np.zeros_like(fp)
            out = np.vstack((fp, out))
        return out[1:]
