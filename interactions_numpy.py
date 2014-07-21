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
    def __init__(self, molecule, protein = False, charge = True):
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
                 ('neighbors', 'float16', (4,3)), # non-H neighbors coordinates for angles (max of 6 neighbors should be enough)
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
        atom_dict = np.empty(molecule.OBMol.NumHvyAtoms(), dtype=atom_dtype)
        i = 0
        for atom in molecule:
            
            atomicnum = atom.atomicnum
            # skip hydrogens for performance
            if atomicnum == 1:
                continue
            atomtype = atom.type
            partialcharge = atom.partialcharge if charge else 0
            coords = atom.coords
            
            if protein:
                residue = pybel.Residue(atom.OBAtom.GetResidue())
            else:
                residue = False
            
            # get neighbors, but only for those atoms which realy need them
            #neighbors = []
            #n_coords = np.empty((6,3), dtype='float16')
            neighbors = np.empty(4, dtype=[('coords', 'float16', 3),('atomicnum', 'int8')])
            neighbors.fill(np.nan)
            n = 0
            for nbr_atom in [x for x in OBAtomAtomIter(atom.OBAtom)]:
                nbr_atomicnum = nbr_atom.GetAtomicNum()
                if nbr_atomicnum == 1:
                    continue
                neighbors[n] = ((nbr_atom.GetX(), nbr_atom.GetY(), nbr_atom.GetZ()), nbr_atomicnum)
                n += 1
            atom_dict[i] = (atom.idx,
                      coords,
                      partialcharge,
                      atomicnum,
                      atomtype,
                      neighbors['coords'], #n_coords,
                      # residue info
                      residue.idx if residue else 0,
                      residue.name if residue else '',
                      residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False, # is backbone
                      # atom properties
                      atom.OBAtom.IsHbondAcceptor(),
                      atom.OBAtom.IsHbondDonor(),
                      atom.OBAtom.IsMetal(),
                      atomicnum == 6 and len(neighbors) > 0 and not (neighbors['atomicnum'] != 6).any(), #hydrophobe
                      atom.OBAtom.IsAromatic(),
                      atomtype in ['O3-', '02-' 'O-'], # is charged (minus)
                      atomtype in ['N3+', 'N2+', 'Ng+'], # is charged (plus)
                      atomicnum in [9,17,35,53], # is halogen?
                      False, # alpha
                      False # beta
                      )
            i +=1
        
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
                path = ring._path
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


def halogenbond_acceptor_halogen(mol1, mol2, base_angle_acceptor = 120, base_angle_halogen = 180, tolerance = 30, cutoff = 4):
    all_a = mol1.atom_dict[mol1.atom_dict['isacceptor']]
    all_h = mol2.atom_dict[mol2.atom_dict['ishalogen']]

    index_crude = np.argwhere(distance(all_a['coords'], all_h['coords']) < cutoff)

    a = all_a[index_crude[:,0]]
    h = all_h[index_crude[:,1]]

    #skip empty values
    if len(a) > 0 and len(h) > 0:
        angle1 = angle_3p(h['coords'],a['coords'],a['neighbors'][:,:,np.newaxis,:])
        angle2 = angle_3p(a['coords'],h['coords'],h['neighbors'][:,:,np.newaxis,:])

        a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]))

        index = np.argwhere(((angle1>(base_angle_acceptor/a_neighbors_num-tolerance)) | np.isnan(angle1)).all(axis=1) & ((angle2>(base_angle_halogen-tolerance)) | np.isnan(angle2)).all(axis=1))  
    
    #halogenbond = np.array([(a,h,False) for i in index])
    return False

def halogenbond(mol1, mol2, base_angle_acceptor = 120, base_angle_halogen = 180, tolerance = 30, cutoff = 4):
    h1 = halogenbond_acceptor_halogen(mol1, mol2, base_angle_acceptor = base_angle_acceptor, base_angle_halogen = base_angle_halogen, tolerance = tolerance, cutoff = cutoff)
    h2 = halogenbond_acceptor_halogen(mol2, mol1, base_angle_acceptor = base_angle_acceptor, base_angle_halogen = base_angle_halogen, tolerance = tolerance, cutoff = cutoff)
    return False


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
    
    if len(m1_plus) > 0 and len(m2_minus) > 0:
        index = np.argwhere(distance(m1_plus['coords'], m2_minus['coords']) < cutoff)
    
    # other way around
    m1_minus = mol1.atom_dict[mol1.atom_dict['isminus']]
    m2_plus = mol2.atom_dict[mol2.atom_dict['isplus']]

    if len(m2_plus) > 0 and len(m1_minus) > 0:
        index = np.argwhere(distance(m2_plus['coords'], m1_minus['coords']) < cutoff)
    
    #salt_bridges
    return False

def hydrophobic_contacts(mol1, mol2, cutoff = 4):
    h1 = mol1.atom_dict[mol1.atom_dict['ishydrophobe']]
    h2 = mol2.atom_dict[mol2.atom_dict['ishydrophobe']]
    
    if len(h1) > 0 and len(h2) > 0:
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

def metal_acceptor(mol1, mol2, base_angle = 120, tolerance = 30, cutoff = 4):
    all_a = mol1.atom_dict[mol1.atom_dict['isacceptor']]
    all_m = mol2.atom_dict[mol2.atom_dict['ismetal']]

    index_crude = np.argwhere(distance(all_a['coords'], all_m['coords']) < cutoff)

    a = all_a[index_crude[:,0]]
    m = all_m[index_crude[:,1]]

    #skip empty values
    if len(a) > 0 and len(m) > 0:
        angle1 = angle_3p(m['coords'],a['coords'],a['neighbors'][:,:,np.newaxis,:])

        a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]))

        index = np.argwhere(((angle1>(base_angle/a_neighbors_num-tolerance)) | np.isnan(angle1)).all(axis=1))
    
    #metalacceptor = np.array([(a,d,False) for i in index])
    #return 
    
        
def metal_pi(mol1, mol2, cutoff = 5, tolerance = 30):
    all_r1 = mol1.ring_dict
    all_m = mol2.atom_dict[mol2.atom_dict['ismetal']]
    
    if len(all_r1) and len(all_m):
        index_crude = np.argwhere(distance(all_r1['centroid'], all_m['coords']) < cutoff)
        
        r1 = all_r1[index_crude[:,0]]
        m = all_m[index_crude[:,1]]
        
        if len(r1) > 0 and len(m) > 0:
            angle1 = angle(r1['vector'], m['coords'] - r1['centroid'])
            index = np.argwhere((angle1 > 180 - tolerance) | (angle1 < tolerance))
    #metalpi
    return False

def metal_coordination(mol1, mol2):
    m1 = metal_acceptor(mol1, mol2)
    m2 = metal_acceptor(mol2, mol1)
    m3 = metal_pi(mol1, mol2)
    m4 = metal_pi(mol2, mol1)
    return False

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

