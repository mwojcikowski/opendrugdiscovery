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

class Molecule:
    def __init__(self, molecule, protein=False):
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
                 ('isdonorh', 'bool'), # realy??
                 ('ismetal', 'bool'),
                 ('ishydrophobe', 'bool'),
                 ('isaromatic', 'bool'),
                 ('ischargedminus', 'bool'),
                 ('ischargedplus', 'bool'),
                 ('ishalogen', 'bool'),
                 # secondary structure
                 ('isalpha', 'bool'),
                 ('isbeta', 'bool')
                 ]

        a = []
        for atom in molecule:
            if protein:
                residue = pybel.Residue(atom.OBAtom.GetResidue())
            else:
                residue = False
            
            # get neighbors
            neighbors = []
            for nbr_atom in [pybel.Atom(x) for x in OBAtomAtomIter(atom.OBAtom)]:
                neighbors.append((nbr_atom.idx,
                                  nbr_atom.coords,
                                  nbr_atom.atomicnum
                                  ))
            neighbors = np.array(neighbors, dtype=[('id', 'int16'),('coords', 'float16', 3),('atomicnum', 'int8')])
            n_coords = np.empty((6,3), dtype='float16')
            n_coords.fill(np.nan)
            n_nonh = neighbors[neighbors['atomicnum']!=1]
            if len(n_nonh) > 0:
                n_coords[:len(n_nonh)] = n_nonh['coords']
            a.append((atom.idx,
                      atom.coords,
                      atom.partialcharge,
                      atom.atomicnum,
                      atom.type,
                      n_coords,
                      # residue info
                      residue.idx if residue else 0,
                      residue.name if residue else 'LIG',
                      residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False, # is backbone
                      # atom properties
                      atom.OBAtom.IsHbondAcceptor(),
                      atom.OBAtom.IsHbondDonor(),
                      atom.OBAtom.IsHbondDonorH(),
                      atom.OBAtom.IsMetal(),
                      atom.atomicnum == 6 and np.in1d(neighbors['atomicnum'], [1,6]).all(), #hydrophobe
                      atom.OBAtom.IsAromatic(),
                      atom.type in ['O3-', '02-' 'O-'], # is charged (minus)
                      atom.type in ['N3+', 'N2+', 'Ng+'], # is charged (plus)
                      atom.atomicnum in [9,17,35,53], # is halogen?
                      False, # alpha
                      False # beta
                      ))
        atom_dict = np.array(a, dtype=atom_dtype)

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
        atom_dict['isalpha'][np.in1d(atom_dict['resid'], res_dict[res_mask_alpha]['id'])] = True

        res_mask_beta = np.where(((phi >= -180) & (phi < -40) & (psi <= 180) & (psi > 90)) | ((phi >= -180) & (phi < -70) & (psi <= -165))) # beta
        res_dict['isbeta'][res_mask_beta] = True
        atom_dict['isbeta'][np.in1d(atom_dict['resid'], res_dict[res_mask_beta]['id'])] = True

        # Rings
        r = []
        for ring in molecule.sssr:
            if ring.IsAromatic():
                path = np.array(ring._path) # path of atom ID's (1-n)
                coords = atom_dict[path-1]['coords']
                centroid = coords.mean(axis=0)
                # get vector perpendicular to ring
                vector = np.cross(atom_dict[path-1]['coords'] - atom_dict[np.hstack((path[1:],path[0]))-1]['coords'],atom_dict[np.hstack((path[1:],path[0:1]))-1]['coords'] - atom_dict[np.hstack((path[2:],path[0:2]))-1]['coords']).mean(axis=0)
                r.append((centroid, vector))
        ring_dict = np.array(r)
        
        self.atom_dict = atom_dict
        self.res_dict = res_dict
        self.ring_dict = ring_dict



def hbond_acceptor_donor(mol1, mol2, cutoff = 3.5, tolerance = 30):
    all_a = mol1.atom_dict[mol1.atom_dict['isacceptor']]
    all_d = mol2.atom_dict[mol2.atom_dict['isdonor']]

    dist = distance(all_a['coords'], all_d['coords'])

    index_crude = np.argwhere(dist < cutoff)

    a = all_a[index_crude[:,0]]
    d = all_d[index_crude[:,1]]

    #skip empty values
    if len(a) == 0 or len(d) == 0:
        return [], []

    angle1 = angle_3p(d['coords'],a['coords'],a['neighbors'][:,:,np.newaxis,:])
    angle2 = angle_3p(a['coords'],d['coords'],d['neighbors'][:,:,np.newaxis,:])

    a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]))
    d_neighbors_num = np.sum(~np.isnan(d['neighbors'][:,:,0]))

    # check for angles (including tolerance) or skip if NaN
    var_a = ((angle1>(120/a_neighbors_num-tolerance)) | np.isnan(angle1)).all(axis=1)
    var_d = ((angle2>(120/d_neighbors_num-tolerance)) | np.isnan(angle2)).all(axis=1)

    index = np.argwhere(var_a & var_d)  
    
    hbond_dtype = [('a_id', 'int16'),
                    ('a_resid', 'int16'),
                    ('a_resname', 'a3'),
                    ('d_id', 'int16'),
                    ('d_resid', 'int16'),
                    ('d_resname', 'a3'),
                    ('isstrict', 'bool')
                    ]
    
    hbond = np.array(np.dstack((a['id'], a['resid'], a['resname'], d['id'], d['resid'], d['resname'])), dtype=hbond_dtype)
    #hbond = np.dstack((a[['id', 'resid', 'resname']], d['id'], d['resid'], d['resname']))
    #hbond = np.concatenate((a[['id', 'resid', 'resname']],a[['id', 'resid', 'resname']]))
    #hbond[:][] = zip(a[index[:,0]],d[index[:,1]])
    return hbond

def hbond(mol1, mol2, cutoff = 3.5, tolerance = 30):
    h1 = hbond_acceptor_donor(mol1, mol2, cutoff = cutoff, tolerance = tolerance)
    h2 = hbond_acceptor_donor(mol2, mol1, cutoff = cutoff, tolerance = tolerance)
    return [h1, h2]



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


def angle(v1, v2):
    """ Return an angle between two vectors in degrees """
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = (v1*v2).sum(axis=-1) # better than np.dot(v1, v2), multiple vectors can be applied
    norm = np.linalg.norm(v1, axis=-1)* np.linalg.norm(v2, axis=-1)
    return np.nan_to_num(np.degrees(np.arccos(dot/norm)))
    
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


