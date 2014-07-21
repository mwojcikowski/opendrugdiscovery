#/usr/bin/python
import numpy as np
from openbabel import OBAtomAtomIter
from scipy.spatial.distance import cdist as distance
#from sklearn.metrics.pairwise import euclidean_distances as distance
import pybel

# define cutoffs
hbond_cutoff = 3.5
hbond_tolerance = 30
salt_bridge_cutoff = 4
hydrophobe_cutoff = 4
pi_cutoff = 5
pi_cation_cutoff = 4
pi_tolerance = 30
halogenbond_cutoff = 4
halogenbond_tolerance = 30

pybel.ob.obErrorLog.StopLogging()

class mol:
    def __init__(self, mol, protein = False):
        self.mol = mol
        self.protein = protein
        self.dict()

        
    def dict(self):
        """ Generate dictionaries of atoms for molecule """

        mol_acceptors = []
        mol_acceptors_id = []
        mol_acceptors_res = []
        mol_acceptors_vec = []

        mol_donors = []
        mol_donors_id = []
        mol_donors_res = []
        mol_donors_vec = []

        mol_salt_plus = []
        mol_salt_plus_id = []
        mol_salt_plus_res = []

        mol_salt_minus = []
        mol_salt_minus_id = []
        mol_salt_minus_res = []

        mol_hydrophobe = []
        mol_hydrophobe_id = []
        mol_hydrophobe_res = []
        
        mol_metal = []
        mol_metal_id = []
        mol_metal_res = []
        
        mol_halogen = []
        mol_halogen_id = []
        mol_halogen_res = []
        mol_halogen_vec = []    
        
        mol_pi = []
        mol_pi_id = []
        mol_pi_vec = []
        mol_pi_res = []
        
        atoms = []
        if self.protein:
            for residue in self.mol.residues:
                for atom in residue:
                    atoms.append((atom, residue))
        else:
            for atom in self.mol:
                atoms.append((atom, None))
        
        if self.protein:
            for atom, residue in atoms:
#            for residue in self.mol.residues:
#                for atom in residue:
                # protein hydrophobe
                if residue.OBResidue.GetAminoAcidProperty(8) and atom.atomicnum == 6:
                    mol_hydrophobe.append(atom.coords)
                    mol_hydrophobe_id.append(atom.idx)
                    if self.protein:
                        mol_hydrophobe_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
            
                # protein hbond
                if atom.OBAtom.IsHbondDonor():
                    mol_donors.append(atom.coords)
                    mol_donors_id.append(atom.idx)
                    if self.protein:
                        mol_donors_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
                    vec = []
                    for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                        if not nbr_atom.OBAtom.IsHbondDonorH():
                            vec.append(np.array(nbr_atom.coords) - np.array(atom.coords))
                    mol_donors_vec.append(vec)
                    # salt
                    if atom.type in ['N3+', 'N2+', 'Ng+']: # types from mol2 and Chimera, ref: http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/idatm.html
                        mol_salt_plus.append(atom.coords)
                        mol_salt_plus_id.append(atom.idx)
                        if self.protein:
                            mol_salt_plus_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
                if atom.OBAtom.IsHbondAcceptor():
                    mol_acceptors.append(atom.coords)
                    mol_acceptors_id.append(atom.idx)
                    if self.protein:
                        mol_acceptors_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
                    vec = []
                    for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                        if not nbr_atom.OBAtom.IsHbondDonorH():
                            vec.append(np.array(nbr_atom.coords) - np.array(atom.coords))
                    mol_acceptors_vec.append(vec)
                    # salt
                    if atom.type in ['O3-', '02-' 'O-']: # types from mol2 and Chimera, ref: http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/idatm.html
                        mol_salt_minus.append(atom.coords)
                        mol_salt_minus_id.append(atom.idx)
                        if self.protein:
                            mol_salt_minus_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))

                # protein metal
                if atom.OBAtom.IsMetal():
                    mol_metal.append(atom.coords)
                    mol_metal_id.append(atom.idx)
                    if self.protein:
                        mol_metal_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
                
                # halogens
                if atom.atomicnum in [9,17,35,53]:
                    mol_halogen.append(atom.coords)
                    mol_halogen_id.append(atom.idx)
                    if self.protein:
                        mol_halogen_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
                    vec = []
                    for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                        vec.append(nbr_atom.coords)
                    mol_halogen_vec.append(vec)
                    

        
            # Pi-rings
            for ring in self.mol.sssr:
                residue = self.mol.atoms[ring._path[0]-1].OBAtom.GetResidue()
                if residue.GetAminoAcidProperty(3): # get aromatic aa
                    ring_coords = []
                    for atom_idx in ring._path:
                        ring_coords.append(self.mol.atoms[atom_idx-1].coords)
                    ring_coords = np.array(ring_coords)
                    ring_center = np.mean(np.array(ring_coords), axis=0)
                    # get mean perpendicular vector to the ring
                    ring_vector = []
                    for i in range(len(ring_coords)):
                        ring_vector.append(np.cross([ring_coords[i] - ring_coords[i-1]],[ring_coords[i-1] - ring_coords[i-2]]))
                    mol_pi.append(ring_center)
                    mol_pi_id.append(ring._path)
                    mol_pi_vec.append(ring_center + np.mean(ring_vector, axis=0))
                    if self.protein:
                        mol_pi_res.append(residue.GetName()[:3]+str(residue.GetNum()))
        # if not protein
        else:
            for atom, residue in atoms:
                # hydrophobe            
                if atom.atomicnum == 6:
                    hydrophobe = True
                    for nbr_atom in OBAtomAtomIter(atom.OBAtom):
                        if nbr_atom.GetAtomicNum() != 6 and nbr_atom.GetAtomicNum() != 1:
                            hydrophobe = False
                            break
                    if hydrophobe:
                        mol_hydrophobe.append(atom.coords)
                        mol_hydrophobe_id.append(atom.idx)
                # hbond
                if atom.OBAtom.IsHbondDonor():
                    mol_donors.append(atom.coords)
                    mol_donors_id.append(atom.idx)
                    vec = []
                    for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                        if not nbr_atom.OBAtom.IsHbondDonorH():
                            vec.append(np.array(nbr_atom.coords) - np.array(atom.coords))
                    mol_donors_vec.append(vec)
                
                    # salt    
                    if atom.type in ['N3+', 'N2+', 'Ng+']: # types from mol2 and Chimera, ref: http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/idatm.html
                        mol_salt_plus.append(atom.coords)
                        mol_salt_plus_id.append(atom.idx)
        
                if atom.OBAtom.IsHbondAcceptor():
                    mol_acceptors.append(atom.coords)
                    mol_acceptors_id.append(atom.idx)
                    vec = []
                    for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                        if not nbr_atom.OBAtom.IsHbondDonorH():
                            vec.append(np.array(nbr_atom.coords) - np.array(atom.coords))
                    mol_acceptors_vec.append(vec)

                    # salt
                    if atom.type in ['O3-', '02-' 'O-']: # types from mol2 and Chimera, ref: http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/idatm.html
                        mol_salt_minus.append(atom.coords)
                        mol_salt_minus_id.append(atom.idx)
                # metals            
                if atom.OBAtom.IsMetal():
                    mol_metal.append(atom.coords)
                    mol_metal_id.append(atom.idx)
            
                # halogens
                if atom.atomicnum in [9,17,35,53]:
                    mol_halogen.append(atom.coords)
                    mol_halogen_id.append(atom.idx)
    #                mol_halogen_res.append(residue.name[:3]+str(residue.OBResidue.GetNum()))
                    vec = []
                    for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                        vec.append(nbr_atom.coords)
                    mol_halogen_vec.append(vec)
                    
            # Pi-rings
            for ring in self.mol.sssr:
                if ring.IsAromatic():
                    ring_coords = []
                    for atom_idx in ring._path:
                        ring_coords.append(self.mol.atoms[atom_idx-1].coords)
                    ring_coords = np.array(ring_coords)
                    ring_center = np.mean(np.array(ring_coords), axis=0)
                    # get mean perpendicular vector to the ring
                    ring_vector = []
                    for i in range(len(ring_coords)):
                        ring_vector.append(np.cross([ring_coords[i] - ring_coords[i-1]],[ring_coords[i-1] - ring_coords[i-2]]))
                    mol_pi.append(ring_center)
                    mol_pi_id.append(ring._path)
                    mol_pi_vec.append(ring_center + np.mean(ring_vector, axis=0))
    #                mol_pi_res.append(residue.GetName()[:3]+str(residue.GetNum()))

        # make dictionaries public
        self.donors = np.array(mol_donors)
        self.donors_id = mol_donors_id
        self.donors_vec = mol_donors_vec
        self.donors_res = mol_donors_res
        
        self.acceptors = np.array(mol_acceptors)
        self.acceptors_id = mol_acceptors_id
        self.acceptors_vec = mol_acceptors_vec
        self.acceptors_res = mol_acceptors_res
        
        self.salt_plus = np.array(mol_salt_plus)
        self.salt_plus_id = mol_salt_plus_id
        self.salt_plus_res = mol_salt_plus_res
        
        self.salt_minus = np.array(mol_salt_minus)
        self.salt_minus_id = mol_salt_minus_id
        self.salt_minus_res = mol_salt_minus_res
        
        self.hydrophobe = np.array(mol_hydrophobe)
        self.hydrophobe_id = mol_hydrophobe_id
        self.hydrophobe_res = mol_hydrophobe_res
        
        self.metal = np.array(mol_metal)
        self.metal_id = mol_metal_id
        self.metal_res = mol_metal_res
        
        self.halogen = np.array(mol_halogen)
        self.halogen_id = mol_halogen_id
        self.halogen_vec = mol_halogen_vec
        self.halogen_res = mol_halogen_res
        
        self.pi = np.array(mol_pi)
        self.pi_id = mol_pi_id
        self.pi_vec = mol_pi_vec
        self.pi_res = mol_pi_res
        



def hbond_acceptor_donor(mol1, mol2):
    hbonds = []
    hbonds_crude = []
    
    if len(mol1.acceptors) > 0 and len(mol2.donors) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.acceptors, mol2.donors) < hbond_cutoff):
            hbond = True # assume that ther is hbond, than check angles
            for v_an in mol1.acceptors_vec[mol1_atom]:
                for v_dn in mol2.donors_vec[mol2_atom]:
                    v_da = mol2.donors[mol2_atom] - mol1.acceptors[mol1_atom]
                    v_ad = mol1.acceptors[mol1_atom] - mol2.donors[mol2_atom]
                    # check if hbond should be discarded
                    if angle(v_an, v_ad) < 120/len(mol1.acceptors_vec[mol1_atom]) - hbond_tolerance or angle(v_da, v_dn) < 120/len(mol2.donors_vec[mol2_atom]) - hbond_tolerance:
                        hbond = False
                        break
                if not hbond:
                    break
            if hbond:
                hbonds.append({'atom_id': [mol1.acceptors_id[mol1_atom],  mol2.donors_id[mol2_atom]], 'res_names': [mol1.acceptors_res[mol1_atom] if mol1.protein == True else '', mol2.donors_res[mol2_atom] if mol2.protein == True else '']})
            else:
                hbonds_crude.append({'atom_id': [mol1.acceptors_id[mol1_atom],  mol2.donors_id[mol2_atom]], 'res_names': [mol1.acceptors_res[mol1_atom] if mol1.protein == True else '', mol2.donors_res[mol2_atom] if mol2.protein == True else '']})

    return hbonds, hbonds_crude

def hbond(mol1, mol2):
    (h1, h1_crude) = hbond_acceptor_donor(mol1, mol2)
    (h2, h2_crude) = hbond_acceptor_donor(mol2, mol1)
    return h1 + h2, h1_crude + h2_crude

def pi_stacking(mol1, mol2):
    pi_stacking = []
    if len(mol1.pi) > 0 and len(mol2.pi) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.pi, mol2.pi) < pi_cutoff):
                v_mol1 = mol1.pi_vec[mol1_atom] - mol1.pi[mol1_atom]
                v_centers = mol2.pi[mol2_atom] - mol1.pi[mol1_atom]
                v_mol2 = mol2.pi_vec[mol2_atom] - mol2.pi[mol2_atom]
                # check angles for face to face, and edge to face
                if (angle(v_mol1, v_centers) < pi_tolerance or angle(v_mol1, v_centers) > 180 - pi_tolerance) and (angle(v_mol2, v_mol1) < pi_tolerance or angle(v_mol2, v_mol1) > 180 - pi_tolerance or np.abs(angle(v_mol2, v_mol1) - 90) < pi_tolerance):
                    pi_stacking.append({'atom_id': [mol1.pi_id[mol1_atom],  mol2.pi_id[mol2_atom]], 'res_names': [mol1.pi_res[mol1_atom] if mol1.protein == True else '', mol2.pi_res[mol2_atom] if mol2.protein == True else '']})
    return pi_stacking

def salt_bridges(mol1, mol2):
    salt_bridges = []
    if len(mol1.salt_minus) > 0 and len(mol2.salt_plus) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.salt_minus, mol2.salt_plus) < salt_bridge_cutoff):
            salt_bridges.append({'atom_id': [mol1.salt_minus_id[mol1_atom],  mol2.salt_plus_id[mol2_atom]], 'res_names': [mol1.salt_minus_res[mol1_atom] if mol1.protein == True else '', mol2.salt_plus_res[mol2_atom] if mol2.protein == True else '']})
    if len(mol1.salt_plus) and len(mol2.salt_minus) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.salt_plus, mol2.salt_minus) < salt_bridge_cutoff):
            salt_bridges.append({'atom_id': [mol1.salt_plus_id[mol1_atom],  mol2.salt_minus_id[mol2_atom]], 'res_names': [mol1.salt_plus_res[mol1_atom] if mol1.protein == True else '', mol2.salt_minus_res[mol2_atom] if mol2.protein == True else '']})
    return salt_bridges

def hydrophobic_contacts(mol1, mol2):
    hydrophobic_contacts = []
    if len(mol1.hydrophobe) > 0 and len(mol2.hydrophobe) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.hydrophobe, mol2.hydrophobe) < hydrophobe_cutoff):
            hydrophobic_contacts.append({'atom_id': [mol1.hydrophobe_id[mol1_atom],  mol2.hydrophobe_id[mol2_atom]], 'res_names': [mol1.hydrophobe_res[mol1_atom] if mol1.protein == True else '', mol2.hydrophobe_res[mol2_atom] if mol2.protein == True else '']})
    return hydrophobic_contacts


def pi_cation(mol1, mol2):
    pi_cation = []
    if len(mol1.salt_plus) > 0 and len(mol2.pi) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.salt_plus, mol2.pi) < pi_cation_cutoff):
            v_pi = mol2.pi_vec[mol2_atom] - mol2.pi[mol2_atom]
            v_cat = mol1.salt_plus[mol1_atom] - mol2.pi[mol2_atom]
            if angle(v_pi, v_cat) < pi_tolerance or angle(v_pi, v_cat) > 180 - pi_tolerance:
                pi_cation.append({'atom_id': [mol1.salt_plus_id[mol1_atom],  mol2.pi_id[mol2_atom]], 'res_names': [mol1.salt_plus_res[mol1_atom] if mol1.protein == True else '', mol2.pi_res[mol2_atom] if mol2.protein == True else '']})
    if len(mol1.pi) > 0 and len(mol2.salt_plus) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.pi, mol2.salt_plus) < pi_cation_cutoff):
            v_pi = mol1.pi_vec[mol1_atom] - mol1.pi[mol1_atom]
            v_cat = mol2.salt_plus[mol2_atom] - mol1.pi[mol1_atom]
            if angle(v_pi, v_cat) < pi_tolerance or angle(v_pi, v_cat) > 180 - pi_tolerance:    
                pi_cation.append({'atom_id': [mol1.pi_id[mol1_atom],  mol2.salt_plus_id[mol2_atom]], 'res_names': [mol1.pi_res[mol1_atom] if mol1.protein == True else '', mol2.salt_plus_res[mol2_atom] if mol2.protein == True else '']})
    return pi_cation


def metal_acceptor(mol1, mol2):
    metal_coordination = []
    metal_coordination_crude = []
    if len(mol1.metal) > 0 and len(mol2.acceptors) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.metal, mol2.acceptors) < hbond_cutoff):
            coord = True # assume that ther is hbond, than check angles
            for v_an in mol2.acceptors_vec[mol2_atom]:
                v_am = mol2.acceptors[mol2_atom] - mol1.metal[mol1_atom]
                # check if hbond should be discarded
                if angle(v_am, v_an) < 120/len(mol2.acceptors_vec[mol2_atom]) - hbond_tolerance:
                    coord = False
                    break    
            if coord:
                metal_coordination.append({'atom_id': [mol1.metal_id[mol1_atom],  mol2.acceptors_id[mol2_atom]], 'res_names': [mol1.metal_res[mol1_atom] if mol1.protein == True else '', mol2.acceptors_res[mol2_atom] if mol2.protein == True else '']})
            else:
                metal_coordination_crude.append({'atom_id': [mol1.metal_id[mol1_atom],  mol2.acceptors_id[mol2_atom]], 'res_names': [mol1.metal_res[mol1_atom] if mol1.protein == True else '', mol2.acceptors_res[mol2_atom] if mol2.protein == True else '']})
    return metal_coordination, metal_coordination_crude
    
        
def metal_pi(mol1, mol2):
    metal_coordination = []
    metal_coordination_crude = []
    if len(mol1.metal) > 0 and len(mol2.pi) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.metal, mol2.pi) < pi_cation_cutoff):
            v_pi = mol2.pi_vec[mol2_atom] - mol2.pi[mol2_atom]
            v_cat = mol1.metal[mol1_atom] - mol2.pi[mol2_atom]
            if angle(v_pi, v_cat) < pi_tolerance or angle(v_pi, v_cat) > 180 - pi_tolerance:
                metal_coordination.append({'atom_id': [mol1.metal_id[mol1_atom],  mol2.pi_id[mol2_atom]], 'res_names': [mol1.metal_res[mol1_atom] if mol1.protein == True else '', mol2.pi_res[mol2_atom] if mol2.protein == True else '']})
            else:
                metal_coordination_crude.append({'atom_id': [mol1.metal_id[mol1_atom],  mol2.pi_id[mol2_atom]], 'res_names': [mol1.metal_res[mol1_atom] if mol1.protein == True else '', mol2.pi_res[mol2_atom] if mol2.protein == True else '']})
    return metal_coordination, metal_coordination_crude

def metal_coordination(mol1, mol2):
    (m1, m1_crude) = metal_acceptor(mol1, mol2)
    (m2, m2_crude) = metal_acceptor(mol2, mol1)
    (m3, m3_crude) = metal_pi(mol1, mol2)
    (m4, m4_crude) = metal_pi(mol2, mol1)
    return m1+m2+m3+m4, m1_crude+m2_crude+m3_crude+m4_crude

def halogenbond_acceptor_halogen(mol1, mol2):
    halogenbonds = []
    halogenbonds_crude = []
    
    if len(mol1.acceptors) > 0 and len(mol2.halogen) > 0:
        for mol1_atom, mol2_atom in np.argwhere(distance(mol1.acceptors, mol2.halogen) < halogenbond_cutoff):
            hbond = True # assume that ther is hydrogenbond, than check angles
            for v_an in mol1.acceptors_vec[mol1_atom]:
                for v_hn in mol2.halogen_vec[mol2_atom]:
                    v_ha = mol2.halogen[mol2_atom] - mol1.acceptors[mol1_atom]
                    v_ah = mol1.acceptors[mol1_atom] - mol2.halogen[mol2_atom]
                    # check if hbond should be discarded
                    if angle(v_an, v_ah) < 120/len(mol1.acceptors_vec[mol1_atom]) - halogenbond_tolerance or angle(v_ha, v_hn) < 150/len(mol2.halogen_vec[mol2_atom]) - halogenbond_tolerance:
                        hbond = False
                        break
                if not hbond:
                    break
            if hbond:
                halogenbonds.append({'atom_id': [mol1.acceptors_id[mol1_atom],  mol2.halogen_id[mol2_atom]], 'res_names': [mol1.acceptors_res[mol1_atom] if mol1.protein == True else '', mol2.halogen_res[mol2_atom] if mol2.protein == True else '']})
            else:
                halogenbonds_crude.append({'atom_id': [mol1.acceptors_id[mol1_atom],  mol2.halogen_id[mol2_atom]], 'res_names': [mol1.acceptors_res[mol1_atom] if mol1.protein == True else '', mol2.halogen_res[mol2_atom] if mol2.protein == True else '']})

    return halogenbonds, halogenbonds_crude

def halogenbond(mol1, mol2):
    (h1, h1_crude) = halogenbond_acceptor_halogen(mol1, mol2)
    (h2, h2_crude) = halogenbond_acceptor_halogen(mol2, mol1)
    return h1 + h2, h1_crude + h2_crude


def angle_3p(p1,p2,p3):
    """ Return an angle from 3 points in cartesian space (point #2 is centroid) """
    v1 = p1-p2
    v2 = p3-p2
    return angle(v1,v2)

def angle(v1, v2):
    """ Return an angle between two vectors in degrees """
    dot = (v1*v2).sum(axis=-1) # better than np.dot(v1, v2), multiple vectors can be applied
    norm = np.linalg.norm(v1, axis=-1)* np.linalg.norm(v2, axis=-1)
    return np.degrees(np.arccos(dot/norm))

def dihedral(p1,p2,p3,p4):
    """ Calculate dihedral from 4 points """
    v12 = (p1-p2)/np.linalg.norm(p1-p2)
    v23 = (p2-p3)/np.linalg.norm(p2-p3)
    v34 = (p3-p4)/np.linalg.norm(p3-p4)
    c1 = np.cross(v12, v23)
    c2 = np.cross(v23, v34)
    out = angle(c1, c2)
    # check clockwise and anti-
    n1 = c1/np.linalg.norm(c1)
    mask = (n1*v34).sum(axis=-1) > 0
    if len(mask.shape) == 0:
        if mask:
            out = -out
    else:
        out[mask] = -out[mask]
    return out


