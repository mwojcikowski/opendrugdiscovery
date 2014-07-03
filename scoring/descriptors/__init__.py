import numpy as np
from scipy.spatial.distance import cdist as distance

class Molecule:
    """ Class which holds dictionaries aiding descriptors calculation """
    def __init__(self, mol):
        """ mol: pybel molecule """
        self.mol = mol
        self._coords = None
        self._charges = None
    
    # lazyload properties and cache them in prefixed [_] variables
    @property
    def coords(self):
        if self._coords is None:
            self._coords = np.array([atom.coords for atom in self.mol])
        return self._coords
    
    @property
    def charges(self):
        if self._charges is None:
            self._charges = np.array([atom.partialcharge for atom in self.mol])
        return self._charges
    
    def coordinate_dict(self, indx_dict):
        """
        Returns vector of atomic coordinates of atoms given by indicies in dictionary 
        
        atomic nums: array of atomic numbers to compute the dictionary 
        """
        mol_atoms = {}
        for key in indx_dict.keys():
            mol_atoms[key] = self.coords[np.array(indx_dict[key], dtype=np.int)-1]
        return mol_atoms
    
    def atom_dict_atomicnum(self, atomic_nums):
        """
        Get dictionary of atom indicies, based on given atom types
        """
        mol_atoms = {}
        for a in atomic_nums:
            mol_atoms[a] = []
        for atom in self.mol:
            if atom.atomicnum in atomic_nums:
                   mol_atoms[atom.atomicnum].append(atom.idx)
        return mol_atoms
    
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
                # SA
                elif atom.atomicnum == 16 and 'SA' in types and atom.OBAtom.IsHbondAcceptor():
                    mol_atoms['SA'].append(atom.coords)
                # ZN
                elif atom.atomicnum == 30 and 'ZN' in types:
                    mol_atoms['ZN'].append(atom.coords)
                    
                    
                    
                    
# DESCRIPTORS

def close_contact(mol1_atoms, mol2_atoms, cutoff):
    """
    Builds descriptor from two dictionaries (protein and ligand) generated by mol_dict()
    """
    desc = []
    for mol2_a in sorted(mol2_atoms.keys()):
        for mol1_a in sorted(mol1_atoms.keys()):
            if len(mol1_atoms[mol1_a])> 0 and len(mol2_atoms[mol2_a]) > 0:
                desc.append(np.sum(distance(mol1_atoms[mol1_a], mol2_atoms[mol2_a]) < cutoff))
            else:
                desc.append(0)
    
    return np.array(desc)
