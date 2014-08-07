import numpy as np
from openbabel import OBAtomAtomIter
import toolkits.ob as toolkit

# hash OB!
toolkit.ob.obErrorLog.StopLogging()

class Molecule(toolkit.Molecule):
    def __init__(self, molecule, protein = False):
        # call parent constructor
        super(Molecule,self).__init__(molecule.OBMol)
        
        self.protein = protein
        
        #ob.DeterminePeptideBackbone(molecule.OBMol)
        # percieve chains in residues
        #if len(res_dict) > 1 and not molecule.OBMol.HasChainsPerceived():
        #    print "Dirty HACK"
        #    molecule = toolkit.readstring('pdb', molecule.write('pdb'))
        
        self._atom_dict = None
        self._res_dict = None
        self._ring_dict = None
        
        
    @property
    def atom_dict(self):
        # check cache and generate dicts
        if self._atom_dict is None:
            self._dicts()
        return self._atom_dict
        
    @property
    def res_dict(self):
        # check cache and generate dicts
        if self._res_dict is None:
            self._dicts()
        return self._res_dict
        
    @property
    def ring_dict(self):
        # check cache and generate dicts
        if self._ring_dict is None:
            self._dicts()
        return self._ring_dict
    
    def _dicts(self):
        # Atoms
        atom_dtype = [('id', 'int16'),
                 # atom info
                 ('coords', 'float16', 3),
                 ('charge', 'float16'),
                 ('atomicnum', 'int8'),
                 ('atomtype','a3'),
                 ('hybridization', 'int8'),
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
        atom_dict = np.empty(self.OBMol.NumHvyAtoms(), dtype=atom_dtype)
        i = 0
        for atom in self:
            
            atomicnum = atom.atomicnum
            # skip hydrogens for performance
            if atomicnum == 1:
                continue
            atomtype = atom.type
            partialcharge = atom.partialcharge
            coords = atom.coords
            
            if self.protein:
                residue = toolkit.Residue(atom.OBAtom.GetResidue())
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
                      atom.OBAtom.GetHyb(),
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
        
        if self.protein:
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
            for residue in self.residues:
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
        for ring in self.sssr:
            if ring.IsAromatic():
                path = ring._path
                coords = atom_dict[np.in1d(atom_dict['id'], path)]['coords']
                centroid = coords.mean(axis=0)
                # get vector perpendicular to ring
                vector = np.cross(coords - np.vstack((coords[1:],coords[:1])), np.vstack((coords[1:],coords[:1])) - np.vstack((coords[2:],coords[:2]))).mean(axis=0) - centroid
                r.append((centroid, vector))
        ring_dict = np.array(r, dtype=[('centroid', 'float16', 3),('vector', 'float16', 3)])
        
        self._atom_dict = atom_dict
        self._ring_dict = ring_dict
        if self.protein:
            self._res_dict = res_dict
            
# angle functions
def angle(p1,p2,p3):
    """ Return an angle from 3 points in cartesian space (point #2 is centroid) """
    v1 = p1-p2
    v2 = p3-p2
    return angle_2v(v1,v2)

def angle_2v(v1, v2):
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
    out = angle_2v(c1, c2)
    # check clockwise and anti-
    n1 = c1/np.linalg.norm(c1)
    mask = (n1*v34).sum(axis=-1) > 0
    if len(mask.shape) == 0:
        if mask:
            out = -out
    else:
        out[mask] = -out[mask]
    return out

