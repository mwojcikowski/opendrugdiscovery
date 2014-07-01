import numpy as np
from scipy.spatial.distance import cdist

def distance(a,b):
    # catch nested (ligands + atomic number groups) coordinates of shape (m-ligands, k-groups, n-atoms, m-dim)
    if len(a.shape) == 3 and len(b.shape) == 4:
        return np.sqrt((b[:,np.newaxis,:,np.newaxis,:,0] - a[:, np.newaxis, :, 0, np.newaxis])**2 + (b[:,np.newaxis,:, np.newaxis,:,1] - a[:, np.newaxis, :, 1, np.newaxis])**2 + (b[:,np.newaxis,:, np.newaxis, :,2] - a[:, np.newaxis, :, 2, np.newaxis])**2)
    # catch nested coordinates of shape (k-groups, n-atoms, m-dim)
    if len(a.shape) == 3 and len(b.shape) == 3:
        return np.sqrt((b[:,np.newaxis,:,0] - a[:, np.newaxis, :, 0, np.newaxis])**2 + (b[:, np.newaxis,:,1] - a[:, np.newaxis, :, 1, np.newaxis])**2 + (b[:, np.newaxis, :,2] - a[:, np.newaxis, :, 2, np.newaxis])**2)
    # otherwise simple group of atom coordinates of shape (n-atoms, m-dim)
    elif len(a.shape) == 2 and len(b.shape) == 2:
        # numpy fastest solution (slower than scipy.spatial.distance.cdist)
        #return np.sqrt(((b[:,0] - a[:,0, np.newaxis])**2 + (b[:,1] - a[:,1, np.newaxis])**2 + (b[:,2] - a[:,2, np.newaxis])**2))
        # scipy
        return cdist(a,b)
    else:
        raise ValueError('Unsuported shape of arrays (%i, %i)' % (len(a.shape), len(b.shape)))

class Molecule:
    """ Class which holds dictionaries aiding descriptors calculation """
    def __init__(self, mol):
        """ mol: pybel molecule """
        self.m = mol
    
    def coordinate_dict(self, atomic_nums):
        """
        Returns vector of atomic coordinates of atoms with atomic number = atomic_num in given molecule 
        
        atomic nums: array of atomic numbers to compute the dictionary 
        """
        mol_atoms = {}
        for a in atomic_nums:
            mol_atoms[a] = []
        for atom in self.m:
            if atom.atomicnum in atomic_nums:
                   mol_atoms[atom.atomicnum].append(atom.coords)
        for a in atomic_nums:
            if len(mol_atoms[a]) > 0:
                mol_atoms[a] = np.array(mol_atoms[a])
            else:
                mol_atoms[a] = np.array([])
        return mol_atoms


# DESCRIPTORS

def close_contact(mol1_atoms, mol2_atoms, cutoff):
    """
    Builds descriptor from two dictionaries (protein and ligand) generated by mol_dict()
    """
    desc = []
    for mol2_a in sorted(mol2_atoms.keys()):
        for mol1_a in sorted(mol1_atoms.keys()):
            if len(mol1_atoms[mol1_a])> 0 and len(mol2_atoms[mol2_a]) > 0:
                desc.append(np.sum(cdist(mol1_atoms[mol1_a], mol2_atoms[mol2_a]) < cutoff))
            else:
                desc.append(0)
    
    return np.array(desc)
