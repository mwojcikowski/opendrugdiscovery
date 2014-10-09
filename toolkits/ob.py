import pybel
from pybel import *
import copy_reg
import numpy as np
from openbabel import OBAtomAtomIter,OBTypeTable
from oddt.spatial import angle, angle_2v, dihedral

backend = 'ob'
# setup typetable to translate atom types
typetable = OBTypeTable()
typetable.SetFromType('INT')
typetable.SetToType('SYB')

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
            self._coords = np.array([atom.coords for atom in self.atoms])
        return self._coords
    
    @property
    def charges(self):
        if self._charges is None:
            self._charges = np.array([atom.partialcharge for atom in self.atoms])
        return self._charges
    
    ### Backport code implementing resudues (by me) to support older versions of OB (aka 'stable')
    @property
    def residue(self): return Residue(self.OBAtom.GetResidue())
    
    #### Custom ODDT properties ####
    def __getattr__(self, attr):
        for desc in pybel._descdict.keys():
            if attr.lower() == desc.lower():
                return self.calcdesc([desc])[desc]
        raise AttributeError('Molecule has no such property: %s' % attr)
    
    @property
    def num_rotors(self):
        return self.OBMol.NumRotors()
    
    def _repr_html_():
        return self.write('svg')
    
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
    
    @property
    def clone(self):
        return Molecule(ob.OBMol(self.OBMol))
    
    def clone_coords(self, source):
        self.OBMol.SetCoordinates(source.OBMol.GetCoordinates())
        return self
    
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
        atom_dict = np.empty(self.OBMol.NumAtoms(), dtype=atom_dtype)
        metals = [3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,39,40,41,42,43,44,45,46,47,48,49,50,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,87,88,89,90,91,
    92,93,94,95,96,97,98,99,100,101,102,103]
        for i, atom in enumerate(self.atoms):
            
            atomicnum = atom.atomicnum
            # skip non-polar hydrogens for performance
#            if atomicnum == 1 and atom.OBAtom.IsNonPolarHydrogen():
#                continue
            atomtype = typetable.Translate(atom.type) # sybyl atom type
            partialcharge = atom.partialcharge
            coords = atom.coords
            
            if self.protein:
                residue = pybel.Residue(atom.OBAtom.GetResidue())
            else:
                residue = False
            
            # get neighbors, but only for those atoms which realy need them
            neighbors = np.empty(4, dtype=[('coords', 'float16', 3),('atomicnum', 'int8')])
            neighbors.fill(np.nan)
            for n, nbr_atom in enumerate(atom.neighbors):
                # concider raising neighbors list to 6, but must do some benchmarks
                if n > 3:
                    break
                nbr_atomicnum = nbr_atom.atomicnum
                neighbors[n] = (nbr_atom.coords, nbr_atomicnum)
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
                      atom.OBAtom.IsHbondDonorH(),
                      atomicnum in metals,
                      atomicnum == 6 and not (np.in1d(neighbors['atomicnum'], [6,1])).any(), #hydrophobe #doble negation, since nan gives False
                      atom.OBAtom.IsAromatic(),
                      atomtype in ['O3-', '02-' 'O-'], # is charged (minus)
                      atomtype in ['N3+', 'N2+', 'Ng+'], # is charged (plus)
                      atomicnum in [9,17,35,53], # is halogen?
                      False, # alpha
                      False # beta
                      )
        
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

### Extend pybel.Molecule
pybel.Molecule = Molecule

class Atom(pybel.Atom):
    @property
    def neighbors(self):
        return [Atom(a) for a in OBAtomAtomIter(self.OBAtom)]
    @property
    def residue(self):
        return Residue(self.OBAtom.GetResidue())

pybel.Atom = Atom

class Residue(object):
    """Represent a Pybel residue.

    Required parameter:
       OBResidue -- an Open Babel OBResidue

    Attributes:
       atoms, idx, name.

    (refer to the Open Babel library documentation for more info).

    The original Open Babel atom can be accessed using the attribute:
       OBResidue
    """

    def __init__(self, OBResidue):
        self.OBResidue = OBResidue

    @property
    def atoms(self):
        return [Atom(atom) for atom in ob.OBResidueAtomIter(self.OBResidue)]

    @property
    def idx(self):
        return self.OBResidue.GetIdx()

    @property
    def name(self):
        return self.OBResidue.GetName()

    def __iter__(self):
        """Iterate over the Atoms of the Residue.

        This allows constructions such as the following:
           for atom in residue:
               print atom
        """
        return iter(self.atoms)


class Fingerprint(pybel.Fingerprint):
    @property
    def raw(self):
        return _unrollbits(self.fp, pybel.ob.OBFingerprint.Getbitsperint())

def _unrollbits(fp, bitsperint):
    """ Unroll unsigned int fingerprint to bool """
    ans = np.zeros(len(fp)*bitsperint)
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
