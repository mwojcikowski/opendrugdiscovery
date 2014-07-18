import numpy as np
from openbabel import OBAtomAtomIter
from scipy.spatial.distance import cdist as distance
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

class Molecule:
    def __init__(self, molecule, protein=False):
        self.protein = protein
        #ob.DeterminePeptideBackbone(molecule.OBMol)

        # percieve chains in residues
        #if len(res_dict) > 1 and not molecule.OBMol.HasChainsPerceived():
        #    print "Dirty HACK"
        #    molecule = pybel.readstring('pdb', molecule.write('pdb'))

        # Atoms
        atom_dtype = [('id', 'int16'),
                 # atom info
                 ('coords', 'float16', 3),
                 ('charge', 'float16'),
                 ('atomicnum', 'int8'),
                 ('atomtype','a3'),
                 ('neighbors', 'float16', (6,3)), # non-H neighbors coordinates for angles (max of 6 neighbors should be enough)
                 # residue info
                 ('resid', 'int16'),
                 ('resname', 'a3'),
                 ('isbackbone', 'bool'),
                 # atom properties
                 ('isacceptor', 'bool'),
                 ('isdonor', 'bool'),
                 ('ismetal', 'bool'),
                 ('ishydrophobe', 'bool'),
                 ('isaromatic', 'bool'),
                 ('isminus', 'bool'),
                 ('isplus', 'bool'),
                 ('ishalogen', 'bool'),
                 # secondary structure
                 ('isalpha', 'bool'),
                 ('isbeta', 'bool')
                 ]

        a = []
        #atom_dict = np.empty(molecule.OBMol.NumAtoms(), dtype=atom_dtype)
        i = 0
        for atom in molecule:
            # skip hydrogens for performance
            if atom.atomicnum == 1:
                continue
            if protein:
                residue = pybel.Residue(atom.OBAtom.GetResidue())
            else:
                residue = False
            
            # get neighbors, but only for those atoms which realy need them
            neighbors = []
            n_coords = np.empty((6,3), dtype='float16')
            for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                if nbr_atom.atomicnum == 1:
                    continue
                neighbors.append((nbr_atom.idx,
                                  nbr_atom.coords,
                                  nbr_atom.atomicnum
                                  ))
            neighbors = np.array(neighbors, dtype=[('id', 'int16'),('coords', 'float16', 3),('atomicnum', 'int8')])
            n_coords.fill(np.nan)
            n_nonh = neighbors#[neighbors['atomicnum']!=1]
            if len(n_nonh) > 0:
                n_coords[:len(n_nonh)] = n_nonh['coords']
            a.append(
            (atom.idx,
                      atom.coords,
                      atom.partialcharge,
                      atom.atomicnum,
                      atom.type,
                      n_coords,
                      # residue info
                      residue.idx if residue else 0,
                      residue.name if residue else '',
                      residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False, # is backbone
                      # atom properties
                      atom.OBAtom.IsHbondAcceptor(),
                      atom.OBAtom.IsHbondDonor(),
                      atom.OBAtom.IsMetal(),
                      atom.atomicnum == 6 and len(neighbors) > 0 and not (neighbors['atomicnum'] != 6).any(), #hydrophobe
                      atom.OBAtom.IsAromatic(),
                      atom.type in ['O3-', '02-' 'O-'], # is charged (minus)
                      atom.type in ['N3+', 'N2+', 'Ng+'], # is charged (plus)
                      atom.atomicnum in [9,17,35,53], # is halogen?
                      False, # alpha
                      False # beta
                      )
            )
            i +=1
        atom_dict = np.array(a, dtype=atom_dtype)
        
        if protein:
            # Protein Residues (alpha helix and beta sheet)
            res_dtype = [('id', 'int16'),
                         ('resname', 'a3'),
                         ('N', 'float16', 3),
                         ('CA', 'float16', 3),
                         ('C', 'float16', 3),
                         ('isalpha', 'bool'),
                         ('isbeta', 'bool')
                         ] # N, CA, C

            b = []
            for residue in molecule.residues:
                backbone = {}
                for atom in residue:
                    if residue.OBResidue.GetAtomProperty(atom.OBAtom,1):
                        if atom.atomicnum == 7:
                            backbone['N'] = atom.coords
                        elif atom.atomicnum == 6:
                            if atom.type == 'C3':
                                backbone['CA'] = atom.coords
                            else:
                                backbone['C'] = atom.coords
                if len(backbone.keys()) == 3:
                    b.append((residue.idx, residue.name, backbone['N'],  backbone['CA'], backbone['C'], False, False))
            res_dict = np.array(b, dtype=res_dtype)
            
            # detect secondary structure by phi and psi angles
            first = res_dict[:-1]
            second = res_dict[1:]
            psi = dihedral(first['N'], first['CA'], first['C'], second['N'])
            phi = dihedral(first['C'], second['N'], second['CA'], second['C'])
            # mark atoms belonging to alpha and beta
            res_mask_alpha = np.where(((phi > -145) & (phi < -35) & (psi > -70) & (psi < 50))) # alpha
            res_dict['isalpha'][res_mask_alpha] = True
            for i in res_dict[res_mask_alpha]['id']:
                atom_dict['isalpha'][atom_dict['resid'] == i] = True

            res_mask_beta = np.where(((phi >= -180) & (phi < -40) & (psi <= 180) & (psi > 90)) | ((phi >= -180) & (phi < -70) & (psi <= -165))) # beta
            res_dict['isbeta'][res_mask_beta] = True
            atom_dict['isbeta'][np.in1d(atom_dict['resid'], res_dict[res_mask_beta]['id'])] = True

        # Aromatic Rings
        r = []
        for ring in molecule.sssr:
            if ring.IsAromatic():
                path = np.array(ring._path) # path of atom ID's (1-n)
                coords = atom_dict[np.in1d(atom_dict['id'], path)]['coords']
                centroid = coords.mean(axis=0)
                # get vector perpendicular to ring
                vector = np.cross(coords - np.vstack((coords[1:],coords[:1])), np.vstack((coords[1:],coords[:1])) - np.vstack((coords[2:],coords[:2]))).mean(axis=0) - centroid
                r.append((centroid, vector))
        ring_dict = np.array(r, dtype=[('centroid', 'float16', 3),('vector', 'float16', 3)])
        
        self.atom_dict = atom_dict
        self.ring_dict = ring_dict
        if protein:
            self.res_dict = res_dict



def hbond_acceptor_donor(mol1, mol2, cutoff = 3.5, base_angle = 120, tolerance = 30):
    all_a = mol1.atom_dict[mol1.atom_dict['isacceptor']]
    all_d = mol2.atom_dict[mol2.atom_dict['isdonor']]

    index_crude = np.argwhere(distance(all_a['coords'], all_d['coords']) < cutoff)

    a = all_a[index_crude[:,0]]
    d = all_d[index_crude[:,1]]

    #skip empty values
    if len(a) > 0 and len(d) > 0:
        angle1 = angle_3p(d['coords'],a['coords'],a['neighbors'][:,:,np.newaxis,:])
        angle2 = angle_3p(a['coords'],d['coords'],d['neighbors'][:,:,np.newaxis,:])

        a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]))
        d_neighbors_num = np.sum(~np.isnan(d['neighbors'][:,:,0]))

        index = np.argwhere(((angle1>(base_angle/a_neighbors_num-tolerance)) | np.isnan(angle1)).all(axis=1) & ((angle2>(base_angle/d_neighbors_num-tolerance)) | np.isnan(angle2)).all(axis=1))  
    
    #hbond = np.array([(a,d,False) for i in index])
    #return 
    
def hbond(mol1, mol2, cutoff = 3.5, tolerance = 30):
    h1 = hbond_acceptor_donor(mol1, mol2, cutoff = cutoff, tolerance = tolerance)
    h2 = hbond_acceptor_donor(mol2, mol1, cutoff = cutoff, tolerance = tolerance)
    #return np.concatenate((h1, h2)), np.concatenate((h1_strict, h2_strict))




def pi_stacking(mol1, mol2, cutoff = 5, tolerance = 30):
    all_r1 = mol1.ring_dict
    all_r2 = mol2.ring_dict
    
    if len(all_r1) > 0 and len(all_r2) > 0:
        index_crude = np.argwhere(distance(all_r1['centroid'], all_r2['centroid']) < cutoff)
        r1 = all_r1[index_crude[:,0]]
        r2 = all_r2[index_crude[:,1]]
        
        if len(r1) == 0 or len(r2) == 0:
            angle1 = angle(r1['vector'],r2['vector'])
            angle2 = angle_3p(r1['vector'] + r1['centroid'],r1['centroid'], r2['centroid'])
            index = np.argwhere(((angle1 > 180 - tolerance) | (angle1 < tolerance) | (angle2 > 90 - tolerance)) & ((angle2 > 180 - tolerance) | (angle1 < tolerance)))
    
    #pistacking
    
    return False

def salt_bridges(mol1, mol2, cutoff = 4):
    m1_plus = mol1.atom_dict[mol1.atom_dict['isplus']]
    m2_minus = mol2.atom_dict[mol2.atom_dict['isminus']]

    index = np.argwhere(distance(m1_plus['coords'], m2_minus['coords']) < cutoff)
    
    # other way around
    m1_minus = mol1.atom_dict[mol1.atom_dict['isminus']]
    m2_plus = mol2.atom_dict[mol2.atom_dict['isplus']]

    index = np.argwhere(distance(m2_plus['coords'], m1_minus['coords']) < cutoff)
    
    #salt_bridges
    return False

def hydrophobic_contacts(mol1, mol2, cutoff = 4):
    h1 = mol1.atom_dict[mol1.atom_dict['ishydrophobe']]
    h2 = mol2.atom_dict[mol2.atom_dict['ishydrophobe']]

    index = np.argwhere(distance(h1['coords'], h2['coords']) < cutoff)
    
    #hydrophobic_contacts
    return False


def pi_cation(mol1, mol2, cutoff = 5, tolerance = 30):
    all_r1 = mol1.ring_dict
    all_plus2 = mol2.atom_dict[mol2.atom_dict['isplus']]
    
    if len(all_r1) and len(all_plus2):
        index_crude = np.argwhere(distance(all_r1['centroid'], all_plus2['coords']) < cutoff)
        
        r1 = all_r1[index_crude[:,0]]
        plus2 = all_plus2[index_crude[:,1]]
        
        if len(r1) > 0 and len(plus2) > 0:
            angle1 = angle(r1['vector'], plus2['coords'] - r1['centroid'])
            index = np.argwhere((angle1 > 180 - tolerance) | (angle1 < tolerance))
    
    # other way around
    all_plus1 = mol1.atom_dict[mol1.atom_dict['isplus']]
    all_r2 = mol2.ring_dict
    
    if len(all_r1) and len(all_plus2):
        index_crude = np.argwhere(distance(all_r2['centroid'], all_plus1['coords']) < cutoff)
        
        plus1 = all_plus1[index_crude[:,1]]
        r2 = all_r2[index_crude[:,0]]
        
        if len(r2) > 0 and len(plus1) > 0:
            angle1 = angle(r1['vector'], plus2['coords'] - r1['centroid'])
            index = np.argwhere((angle1 > 180 - tolerance) | (angle1 < tolerance))
    
    #pication
    
    return False








#######################################################################333



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





