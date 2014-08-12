import numpy as np
from scipy.spatial.distance import cdist as distance

class close_contacts:
    def __init__(self, protein = None, cutoff = 4, mode = 'atomic_nums', ligand_types = None, protein_types = None):
        self.cutoff = cutoff
        self.protein = protein
        self.ligand_types = ligand_types
        self.protein_types = protein_types if protein_types else ligand_types
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
        
        return False
    
    def _atom_types_sybyl(self, atom_dict, protein = False):
        
        return False
    
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
        
        
        
        
        
        
#    def atom_dict_types(self, types, mode='ad4'):
#        """
#        Get dictionary of atom indicies, based on given atom types
#        types: an array of types as strings
#        mode: which types to use (ad4, sybyl)
#        """
#        mol_atoms = {}
#        for t in types:
#            mol_atoms[t] = []
#        if mode == 'ad4': # AutoDock4 types http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters
#            for atom in self.mol:
#                # A
#                if atom.type == 'Car' and 'A' in types:
#                    mol_atoms['A'].append(atom.coords)
#                # C
#                elif atom.atomicnum == 6 and 'C' in types:
#                    mol_atoms['C'].append(atom.coords)
#                # CL
#                elif atom.atomicnum == 17 and 'CL' in types:
#                    mol_atoms['CL'].append(atom.coords)
#                # F
#                elif atom.atomicnum == 9 and 'F' in types:
#                    mol_atoms['F'].append(atom.coords)
#                # FE
#                elif atom.atomicnum == 26 and 'FE' in types:
#                    mol_atoms['FE'].append(atom.coords)
#                # HD
#                elif atom.atomicnum == 1 and 'HD' in types and atom.OBAtom.IsHbondDonorH():
#                    mol_atoms['HD'].append(atom.coords)
#                # MG
#                elif atom.atomicnum == 12 and 'MG' in types:
#                    mol_atoms['MG'].append(atom.coords)
#                # MN
#                elif atom.atomicnum == 12 and 'MN' in types:
#                    mol_atoms['MN'].append(atom.coords)
#                # NA
#                elif atom.atomicnum == 7 and 'NA' in types and atom.OBAtom.IsHbondAcceptor():
#                    mol_atoms['NA'].append(atom.coords)
#                # N
#                elif atom.atomicnum == 7 and 'N' in types:
#                    mol_atoms['N'].append(atom.coords)
#                # OA
#                elif atom.atomicnum == 8 and 'OA' in types and atom.OBAtom.IsHbondAcceptor():
#                    mol_atoms['OA'].append(atom.coords)
#                # O
#                elif atom.atomicnum == 8 and 'O' in types:
#                    mol_atoms['O'].append(atom.coords)
#                # SA
#                elif atom.atomicnum == 16 and 'SA' in types and atom.OBAtom.IsHbondAcceptor():
#                    mol_atoms['SA'].append(atom.coords)
#                # S
#                elif atom.atomicnum == 16 and 'S' in types:
#                    mol_atoms['S'].append(atom.coords)
#                # ZN
#                elif atom.atomicnum == 30 and 'ZN' in types:
#                    mol_atoms['ZN'].append(atom.coords)

