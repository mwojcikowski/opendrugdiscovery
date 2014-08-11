import numpy as np
import pybel
from scipy.spatial.distance import cdist as distance

from ....interactions import close_contacts as inter_close_contacts

    
    def atom_dict_types(self, types, mode='ad4'):
        """
        Get dictionary of atom indicies, based on given atom types
        types: an array of types as strings
        mode: which types to use (ad4, sybyl)
        """
        mol_atoms = {}
        for t in types:
            mol_atoms[t] = []
        if mode == 'ad4': # AutoDock4 types http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters
            for atom in self.mol:
                # A
                if atom.type == 'Car' and 'A' in types:
                    mol_atoms['A'].append(atom.coords)
                # C
                elif atom.atomicnum == 6 and 'C' in types:
                    mol_atoms['C'].append(atom.coords)
                # CL
                elif atom.atomicnum == 17 and 'CL' in types:
                    mol_atoms['CL'].append(atom.coords)
                # F
                elif atom.atomicnum == 9 and 'F' in types:
                    mol_atoms['F'].append(atom.coords)
                # FE
                elif atom.atomicnum == 26 and 'FE' in types:
                    mol_atoms['FE'].append(atom.coords)
                # HD
                elif atom.atomicnum == 1 and 'HD' in types and atom.OBAtom.IsHbondDonorH():
                    mol_atoms['HD'].append(atom.coords)
                # MG
                elif atom.atomicnum == 12 and 'MG' in types:
                    mol_atoms['MG'].append(atom.coords)
                # MN
                elif atom.atomicnum == 12 and 'MN' in types:
                    mol_atoms['MN'].append(atom.coords)
                # NA
                elif atom.atomicnum == 7 and 'NA' in types and atom.OBAtom.IsHbondAcceptor():
                    mol_atoms['NA'].append(atom.coords)
                # N
                elif atom.atomicnum == 7 and 'N' in types:
                    mol_atoms['N'].append(atom.coords)
                # OA
                elif atom.atomicnum == 8 and 'OA' in types and atom.OBAtom.IsHbondAcceptor():
                    mol_atoms['OA'].append(atom.coords)
                # O
                elif atom.atomicnum == 8 and 'O' in types:
                    mol_atoms['O'].append(atom.coords)
                # SA
                elif atom.atomicnum == 16 and 'SA' in types and atom.OBAtom.IsHbondAcceptor():
                    mol_atoms['SA'].append(atom.coords)
                # S
                elif atom.atomicnum == 16 and 'S' in types:
                    mol_atoms['S'].append(atom.coords)
                # ZN
                elif atom.atomicnum == 30 and 'ZN' in types:
                    mol_atoms['ZN'].append(atom.coords)
                    
                    
                    
                    
# PROTEIN-LIGAND DESCRIPTORS
def close_contacts(mol1, mol2, cutoff):
    contacts = inter_close_contacts
    desc = []
    for mol2_a in sorted(mol2_atoms.keys()):
        for mol1_a in sorted(mol1_atoms.keys()):
            if len(mol1_atoms[mol1_a])> 0 and len(mol2_atoms[mol2_a]) > 0:
                desc.append(np.sum(distance(mol1_atoms[mol1_a], mol2_atoms[mol2_a]) < cutoff))
            else:
                desc.append(0)
    
    return np.array(desc)

class close_contacts:
    def __init__(self, protein, mode = 'atomic_nums', ligand_types = None, protein_types = None):
        self.protein = protein
        self.ligand_types = ligand_types
        self.protein_types = protein_types if protein_types else ligand_types
        if mode == 'atomic_nums':
            self.typer = 
        elif mode == 'atomic_types_sybyl':
            self.typer = 
        elif mode == 'atomic_types_ad4':
            self.typer = 
    
    def build(self, ligands):
        for mol in ligands:
            

class fingerprints:
    def __init__(self, fp = 'fp2', toolkit = 'ob'):
        self.fp = fp
        if toolkit == oddt.default_toolkit:
            self.exchange = False
        else:
            self.exchange = True
            self.target_toolkit = __import__('toolkits.'+toolkit)
    
    def _get_fingerptint(self, mol)
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
