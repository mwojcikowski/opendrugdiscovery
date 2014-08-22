import numpy as np
from oddt.spatial import dihedral, angle, angle_2v, distance

def close_contacts(x, y, cutoff, x_column = 'coords', y_column = 'coords'):
    """ Returns pairs of close contacts atoms within cutoff. """
    if len(x[x_column]) > 0 and len(x[x_column]) > 0:
        index = np.argwhere(distance(x[x_column], y[y_column]) < cutoff)
        return x[index[:,0]], y[index[:,1]]
    else:
        return x[[]], y[[]]

def hbond_acceptor_donor(mol1, mol2, cutoff = 3.5, base_angle = 120, tolerance = 30):
    a, d = close_contacts(mol1.atom_dict[mol1.atom_dict['isacceptor']], mol2.atom_dict[mol2.atom_dict['isdonor']], cutoff)
    #skip empty values
    if len(a) > 0 and len(d) > 0:
        angle1 = angle(d['coords'][:,np.newaxis,:],a['coords'][:,np.newaxis,:],a['neighbors'])
        angle2 = angle(a['coords'][:,np.newaxis,:],d['coords'][:,np.newaxis,:],d['neighbors'])
        a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]), axis=-1)[:,np.newaxis]
        d_neighbors_num = np.sum(~np.isnan(d['neighbors'][:,:,0]), axis=-1)[:,np.newaxis]
        strict = (((angle1>(base_angle/a_neighbors_num-tolerance)) | np.isnan(angle1)) & ((angle2>(base_angle/d_neighbors_num-tolerance)) | np.isnan(angle2))).all(axis=-1)
        return a, d, strict
    else:
        return a, d, np.array([], dtype=bool) 
    
def hbond(mol1, mol2, cutoff = 3.5, tolerance = 30):
    a1, d1, s1 = hbond_acceptor_donor(mol1, mol2, cutoff = cutoff, tolerance = tolerance)
    a2, d2, s2 = hbond_acceptor_donor(mol2, mol1, cutoff = cutoff, tolerance = tolerance)
    return np.concatenate((a1, d2)), np.concatenate((d1, a2)), np.concatenate((s1, s2))

def halogenbond_acceptor_halogen(mol1, mol2, base_angle_acceptor = 120, base_angle_halogen = 180, tolerance = 30, cutoff = 4):
    a, h = close_contacts(mol1.atom_dict[mol1.atom_dict['isacceptor']], mol2.atom_dict[mol2.atom_dict['ishalogen']], cutoff)
    #skip empty values
    if len(a) > 0 and len(h) > 0:
        angle1 = angle(h['coords'][:,np.newaxis,:],a['coords'][:,np.newaxis,:],a['neighbors'])
        angle2 = angle(a['coords'][:,np.newaxis,:],h['coords'][:,np.newaxis,:],h['neighbors'])
        a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]), axis=-1)[:,np.newaxis]
        h_neighbors_num = np.sum(~np.isnan(h['neighbors'][:,:,0]), axis=-1)[:,np.newaxis]
        strict = (((angle1>(base_angle_acceptor/a_neighbors_num-tolerance)) | np.isnan(angle1)) & ((angle2>(base_angle_halogen/h_neighbors_num-tolerance)) | np.isnan(angle2))).all(axis=-1)
        return a, h, strict
    else:
        return a, h, np.array([], dtype=bool)

def halogenbond(mol1, mol2, base_angle_acceptor = 120, base_angle_halogen = 180, tolerance = 30, cutoff = 4):
    a1, h1, s1 = halogenbond_acceptor_halogen(mol1, mol2, base_angle_acceptor = base_angle_acceptor, base_angle_halogen = base_angle_halogen, tolerance = tolerance, cutoff = cutoff)
    a2, h2, s2 = halogenbond_acceptor_halogen(mol2, mol1, base_angle_acceptor = base_angle_acceptor, base_angle_halogen = base_angle_halogen, tolerance = tolerance, cutoff = cutoff)
    return np.concatenate((a1, h2)), np.concatenate((h1, a2)), np.concatenate((s1, s2))

def pi_stacking(mol1, mol2, cutoff = 5, tolerance = 30):
    r1, r2 = close_contacts(mol1.ring_dict, mol2.ring_dict, cutoff, x_column = 'centroid', y_column = 'centroid')
    if len(r1) > 0 and len(r2) > 0:
        angle1 = angle_2v(r1['vector'],r2['vector'])
        angle2 = angle(r1['vector'] + r1['centroid'],r1['centroid'], r2['centroid'])
        strict_paralel = ((angle1 > 180 - tolerance) | (angle1 < tolerance)) & ((angle2 > 180 - tolerance) | (angle2 < tolerance))
        strict_perpendicular = ((angle1 > 90 - tolerance) & (angle1 < 90 + tolerance)) & ((angle2 > 180 - tolerance) | (angle2 < tolerance))
        return r1, r2, strict_paralel, strict_perpendicular
    else:
        return r1, r2, np.array([], dtype=bool), np.array([], dtype=bool)

def salt_bridge_plus_minus(mol1, mol2, cutoff = 4):
    m1_plus, m2_minus = close_contacts(mol1.atom_dict[mol1.atom_dict['isplus']], mol2.atom_dict[mol2.atom_dict['isminus']], cutoff)
    return m1_plus, m2_minus

def salt_bridges(mol1, mol2, cutoff = 4):
    m1_plus, m2_minus = salt_bridge_plus_minus(mol1, mol2, cutoff = cutoff)
    m2_plus, m1_minus = salt_bridge_plus_minus(mol2, mol1, cutoff = cutoff)
    return np.concatenate((m1_plus, m1_minus)), np.concatenate((m2_minus, m2_plus))

def hydrophobic_contacts(mol1, mol2, cutoff = 4):
    h1, h2 = close_contacts(mol1.atom_dict[mol1.atom_dict['ishydrophobe']], mol2.atom_dict[mol2.atom_dict['ishydrophobe']], cutoff)
    return h1, h2

def pi_cation(mol1, mol2, cutoff = 5, tolerance = 30):
    r1, plus2 = close_contacts(mol1.ring_dict, mol2.atom_dict[mol2.atom_dict['isplus']], cutoff, x_column='centroid')
    if len(r1) > 0 and len(plus2) > 0:
        angle1 = angle_2v(r1['vector'], plus2['coords'] - r1['centroid'])
        strict = (angle1 > 180 - tolerance) | (angle1 < tolerance)
        return r1, plus2, strict
    else:
        return r1, plus2, np.array([], dtype=bool)

def acceptor_metal(mol1, mol2, base_angle = 120, tolerance = 30, cutoff = 4):
    a, m = close_contacts(mol1.atom_dict[mol1.atom_dict['isacceptor']], mol2.atom_dict[mol2.atom_dict['ismetal']], cutoff)
    #skip empty values
    if len(a) > 0 and len(m) > 0:
        angle1 = angle(m['coords'][:,np.newaxis,:],a['coords'][:,np.newaxis,:],a['neighbors'])
        a_neighbors_num = np.sum(~np.isnan(a['neighbors'][:,:,0]), axis=-1)[:,np.newaxis]
        strict = ((angle1>(base_angle/a_neighbors_num-tolerance)) | np.isnan(angle1)).all(axis=-1)
        return a, m, strict
    else:
        return a, m, np.array([], dtype=bool)

def pi_metal(mol1, mol2, cutoff = 5, tolerance = 30):
    r1, m = close_contacts(mol1.ring_dict, mol2.atom_dict[mol2.atom_dict['ismetal']], cutoff, x_column='centroid')
    if len(r1) > 0 and len(m) > 0:
        angle1 = angle_2v(r1['vector'], m['coords'] - r1['centroid'])
        strict = (angle1 > 180 - tolerance) | (angle1 < tolerance)
        return r1, m, strict
    else:
        return r1, m, np.array([], dtype=bool)

## TODO # probably delete it, just for benchmarking' sake
def metal_coordination(mol1, mol2):
    m1 = acceptor_metal(mol1, mol2)
    m2 = acceptor_metal(mol2, mol1)
    m3 = pi_metal(mol1, mol2)
    m4 = pi_metal(mol2, mol1)
    return False
