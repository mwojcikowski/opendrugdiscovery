import numpy as np
from cinfony import rdk
from cinfony.rdk import *
from rdkit.Chem.Lipinski import NumRotatableBonds

class Molecule(rdk.Molecule):
    def __init__(self, Mol, protein = False):
        # call parent constructor
        super(Molecule,self).__init__(Mol)
        
        self.protein = protein
        
        self._atom_dict = None
        self._res_dict = None
        self._ring_dict = None
        self._coords = None
        self._charges = None
        
    # cache frequently used properties and cache them in prefixed [_] variables
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
    
    #### Custom ODDT properties ####
    @property
    def num_rotors(self):
        return NumRotatableBonds(self.Mol)
    
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
                 ('atomtype','a4'),
                 ('hybridization', 'int8'),
                 ('neighbors', 'float16', (4,3)), # non-H neighbors coordinates for angles (max of 6 neighbors should be enough)
                 # residue info
                 ('resid', 'int16'),
                 ('resname', 'a3'),
                 ('isbackbone', 'bool'),
                 # atom properties
                 ('isacceptor', 'bool'),
                 ('isdonor', 'bool'),
                 ('isdonorh', 'bool'),
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
        atom_dict = np.empty(self.Mol.GetNumAtoms(), dtype=atom_dtype)
        i = 0
        for atom in self.atoms:
            
            atomicnum = atom.atomicnum
            partialcharge = atom.partialcharge ## TODO: gasteiger charges
            coords = atom.coords
            
            if self.protein:
                residue = pybel.Residue(atom.OBAtom.GetResidue())
            else:
                residue = False
            
            # get neighbors, but only for those atoms which realy need them
            neighbors = np.empty(4, dtype=[('coords', 'float16', 3),('atomicnum', 'int8')])
            neighbors.fill(np.nan)
            for n, nbr_atom in enumerate(atom.neighbors):
                neighbors[n] = (nbr_atom.coords, nbr_atom.atomicnum)
            atom_dict[i] = (atom.idx,
                      coords,
                      partialcharge,
                      atomicnum,
                      atomtype,
                      atom.Atom.GetHybridization(), ######################################## tu wstepnie zrobione
                      neighbors['coords'], #n_coords,
                      # residue info
                      residue.idx if residue else 0,
                      residue.name if residue else '',
                      residue.OBResidue.GetAtomProperty(atom.OBAtom, 2) if residue else False, # is backbone
                      # atom properties
                      atom.OBAtom.IsHbondAcceptor(),
                      atom.OBAtom.IsHbondDonor(),
                      atom.OBAtom.IsHbondDonorH(),
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
                atom = atom_dict[atom_dict['id'] == path[0]]
                coords = atom_dict[np.in1d(atom_dict['id'], path)]['coords']
                centroid = coords.mean(axis=0)
                # get vector perpendicular to ring
                vector = np.cross(coords - np.vstack((coords[1:],coords[:1])), np.vstack((coords[1:],coords[:1])) - np.vstack((coords[2:],coords[:2]))).mean(axis=0) - centroid
                r.append((centroid, vector, atom['isalpha'], atom['isbeta']))
        ring_dict = np.array(r, dtype=[('centroid', 'float16', 3),('vector', 'float16', 3),('isalpha', 'bool'),('isbeta', 'bool'),])
        
        self._atom_dict = atom_dict
        self._ring_dict = ring_dict
        if self.protein:
            self._res_dict = res_dict
    
    
rdk.Molecule = Molecule

class Fingerprint(rdk.Fingerprint):
    @property
    def raw(self):
        return np.array(self.fp, dtype=bool)
        
rdk.Fingerprint = Fingerprint

# Patch reader not to return None as molecules
def _readfile(format, filename):
    for mol in rdk.readfile(format, filename):
        if mol is not None:
            yield mol

rdk.readfile = _readfile
