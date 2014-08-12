import pybel
from pybel import *
import copy_reg
import numpy as np
from openbabel import OBAtomAtomIter

from .. import angle,angle_2v,dihedral

backend = 'ob'

# hash OB!
pybel.ob.obErrorLog.StopLogging()

class Molecule(pybel.Molecule):
    def __init__(self, OBMol, protein = False):
        # call parent constructor
        super(Molecule,self).__init__(OBMol)
        
        self.protein = protein
        
        #ob.DeterminePeptideBackbone(molecule.OBMol)
        # percieve chains in residues
        #if len(res_dict) > 1 and not molecule.OBMol.HasChainsPerceived():
        #    print "Dirty HACK"
        #    molecule = pybel.readstring('pdb', molecule.write('pdb'))
        
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

### Extend pybel.Molecule
pybel.Molecule = Molecule

class Fingerprint(pybel.Fingerprint):
    @property
    def raw(self):
        return _unrollbits(self.fp, pybel.ob.OBFingerprint.Getbitsperint())

def _unrollbits(fp, bitsperint):
    """ Unroll unsigned int fingerprint to bool """
    ans = np.zeros(len(fp)*bitsperint, dtype=bool)
    start = 1
    for x in fp:
        i = start
        while x > 0:
            ans[i] = x % 2
            x >>= 1
            i += 1
        start += bitsperint
    return ans

pybel.Fingerprint = Fingerprint

### Monkeypatch pybel objects pickling
pickle_format = 'mol2'
def pickle_mol(self):
    return unpickle_mol, (self.write(pickle_format), dict(self.data.items()))

def unpickle_mol(string, data):
    mol = readstring(pickle_format, string)
    mol.data.update(data)
    return mol
copy_reg.pickle(Molecule, pickle_mol, unpickle_mol)
