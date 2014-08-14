import numpy as np
from scipy.spatial.distance import cdist as distance

def atoms_by_type(atom_dict, types, mode = 'atomic_nums'):
    """
    AutoDock4 types definition: http://autodock.scripps.edu/faqs-help/faq/where-do-i-set-the-autodock-4-force-field-parameters
    """
    if mode == 'atomic_nums':
        return {num: atom_dict[atom_dict['atomicnum'] == num] for num in set(types)}
    elif mode == 'atom_types_sybyl':
        return {t: atom_dict[atom_dict['atomtype'] == t] for t in set(types)}
    elif mode == 'atom_types_ad4':
        # all AD4 atom types are capitalized
        types = [t.upper() for t in types]
        out = {}
        for t in set(types):
            if t == 'HD':
                out[t] = atom_dict[atom_dict['atomicnum'] == 1 & atom_dict['isdonorh']]
            elif t == 'C':
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & ~atom_dict['isaromatic']]
            elif t == 'CD': # not canonical AD4 type, although used by NNscore, with no description. properies assued by name
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & ~atom_dict['isdonor']]
            elif t == 'A':
                out[t] = atom_dict[atom_dict['atomicnum'] == 6 & atom_dict['isaromatic']]
            elif t == 'N':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & ~atom_dict['isacceptor']]
            elif t == 'NA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 7 & atom_dict['isacceptor']]
            elif t == 'OA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 8 & atom_dict['isacceptor']]
            elif t == 'F':
                out[t] = atom_dict[atom_dict['atomicnum'] == 9]
            elif t == 'MG':
                out[t] = atom_dict[atom_dict['atomicnum'] == 12]
            elif t == 'P':
                out[t] = atom_dict[atom_dict['atomicnum'] == 15]
            elif t == 'SA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 16 & atom_dict['isacceptor']]
            elif t == 'S':
                out[t] = atom_dict[atom_dict['atomicnum'] == 16 & ~atom_dict['isacceptor']]
            elif t == 'CL':
                out[t] = atom_dict[atom_dict['atomicnum'] == 17]
            elif t == 'CA':
                out[t] = atom_dict[atom_dict['atomicnum'] == 20]
            elif t == 'MN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 25]
            elif t == 'FE':
                out[t] = atom_dict[atom_dict['atomicnum'] == 26]
            elif t == 'CU':
                out[t] = atom_dict[atom_dict['atomicnum'] == 29]
            elif t == 'ZN':
                out[t] = atom_dict[atom_dict['atomicnum'] == 30]
            elif t == 'BR':
                out[t] = atom_dict[atom_dict['atomicnum'] == 35]
            elif t == 'I':
                out[t] = atom_dict[atom_dict['atomicnum'] == 53]
            else:
                 raise ValueError('Unsopported atom type: %s' % t)
        return out

class close_contacts(object):
    def __init__(self, protein = None, cutoff = 4, mode = 'atomic_nums', ligand_types = None, protein_types = None, aligned_pairs = False):
        self.cutoff = cutoff
        self.protein = protein
        self.ligand_types = ligand_types
        self.protein_types = protein_types if protein_types else ligand_types
        self.aligned_pairs = aligned_pairs
        self.mode = mode
    
    def build(self, ligands, protein = None, single = False):
        if protein is None:
            protein = self.protein
        if single:
            ligands = [ligands]
        prot_dict = atoms_by_type(protein.atom_dict, self.protein_types, self.mode)
        desc_size = len(self.ligand_types) if self.aligned_pairs else len(self.ligand_types)*len(self.protein_types)
        out = np.zeros(desc_size, dtype=int)
        for mol in ligands:
            mol_dict = atoms_by_type(mol.atom_dict, self.ligand_types, self.mode) 
            if self.aligned_pairs:
                desc = np.array([(distance(prot_dict[prot_type]['coords'], mol_dict[mol_type]['coords']) <= self.cutoff).sum() for mol_type, prot_type in zip(self.ligand_types, self.protein_types)], dtype=int)
            else:
                desc = np.array([(distance(prot_dict[prot_type]['coords'], mol_dict[mol_type]['coords']) <= self.cutoff).sum() for prot_type in self.protein_types for mol_type in self.ligand_types], dtype=int)
            out = np.vstack((out, desc))
        return out[1:]
        
class fingerprints(object):
    def __init__(self, fp = 'fp2', toolkit = 'ob'):
        self.fp = fp
        self.exchange = False
        #if toolkit == oddt.toolkit.backend:
        #    self.exchange = False
        #else:
        #    self.exchange = True
        #    self.target_toolkit = __import__('toolkits.'+toolkit)
    
    def _get_fingerprint(self, mol):
        if self.exchange:
            mol = self.target_toolkit.Molecule(mol)
        return mol.calcfp(self.fp).raw
        
    def build(self, mols, single = False):
        if single:
            mols = [mols]
        out = None
        
        for mol in mols:
            fp = self._get_fingerprint(mol)
            if out is None:
                out = np.zeros_like(fp)
            out = np.vstack((fp, out))
        return out[1:]
        
from oddt.docking.autodock_vina import autodock_vina
from oddt.scoring.descriptors import atoms_by_type, close_contacts
from oddt import interactions

class binana_descriptor:
    def __init__(self, protein):
        self.protein = protein
        self.vina = autodock_vina(protein)
        # Close contacts descriptor generators
        cc_4_types = (('A', 'A'), ('A', 'C'), ('A', 'CL'), ('A', 'F'), ('A', 'FE'), ('A', 'HD'), ('A', 'MG'), ('A', 'MN'), ('A', 'N'), ('A', 'NA'), ('A', 'OA'), ('A', 'SA'), ('A', 'ZN'), ('BR', 'C'), ('BR', 'HD'), ('BR', 'OA'), ('C', 'C'), ('C', 'CL'), ('C', 'F'), ('C', 'HD'), ('C', 'MG'), ('C', 'MN'), ('C', 'N'), ('C', 'NA'), ('C', 'OA'), ('C', 'SA'), ('C', 'ZN'), ('CL', 'FE'), ('CL', 'HD'), ('CL', 'MG'), ('CL', 'N'), ('CL', 'OA'), ('CL', 'ZN'), ('F', 'HD'), ('F', 'N'), ('F', 'OA'), ('F', 'SA'), ('FE', 'HD'), ('FE', 'N'), ('FE', 'OA'), ('HD', 'HD'), ('HD', 'I'), ('HD', 'MG'), ('HD', 'MN'), ('HD', 'N'), ('HD', 'NA'), ('HD', 'OA'), ('HD', 'P'), ('HD', 'S'), ('HD', 'SA'), ('HD', 'ZN'), ('MG', 'NA'), ('MG', 'OA'), ('MN', 'N'), ('MN', 'OA'), ('N', 'N'), ('N', 'NA'), ('N', 'OA'), ('N', 'SA'), ('N', 'ZN'), ('NA', 'OA'), ('NA', 'SA'), ('NA', 'ZN'), ('OA', 'OA'), ('OA', 'SA'), ('OA', 'ZN'), ('S', 'ZN'), ('SA', 'ZN'), ('A', 'BR'), ('A', 'I'), ('A', 'P'), ('A', 'S'), ('BR', 'N'), ('BR', 'SA'), ('C', 'FE'), ('C', 'I'), ('C', 'P'), ('C', 'S'), ('CL', 'MN'), ('CL', 'NA'), ('CL', 'P'), ('CL', 'S'), ('CL', 'SA'), ('CU', 'HD'), ('CU', 'N'), ('FE', 'NA'), ('FE', 'SA'), ('I', 'N'), ('I', 'OA'), ('MG', 'N'), ('MG', 'P'), ('MG', 'S'), ('MG', 'SA'), ('MN', 'NA'), ('MN', 'P'), ('MN', 'S'), ('MN', 'SA'), ('N', 'P'), ('N', 'S'), ('NA', 'P'), ('NA', 'S'), ('OA', 'P'), ('OA', 'S'), ('P', 'S'), ('P', 'SA'), ('P', 'ZN'), ('S', 'SA'), ('SA', 'SA'), ('A', 'CU'), ('C', 'CD') )
        cc_4_rec_types, cc_4_lig_types = zip(*cc_4_types)
        self.cc_4 = cc_4_nn = close_contacts(protein, cutoff=4, protein_types=cc_4_rec_types, ligand_types=cc_4_lig_types, mode='atom_types_ad4', aligned_pairs=True)
        cc_25_types = [('A', 'A'), ('A', 'C'), ('A', 'CL'), ('A', 'F'), ('A', 'FE'), ('A', 'HD'), ('A', 'MG'), ('A', 'MN'), ('A', 'N'), ('A', 'NA'), ('A', 'OA'), ('A', 'SA'), ('A', 'ZN'), ('BR', 'C'), ('BR', 'HD'), ('BR', 'OA'), ('C', 'C'), ('C', 'CL'), ('C', 'F'), ('C', 'HD'), ('C', 'MG'), ('C', 'MN'), ('C', 'N'), ('C', 'NA'), ('C', 'OA'), ('C', 'SA'), ('C', 'ZN'), ('CD', 'OA'), ('CL', 'FE'), ('CL', 'HD'), ('CL', 'MG'), ('CL', 'N'), ('CL', 'OA'), ('CL', 'ZN'), ('F', 'HD'), ('F', 'N'), ('F', 'OA'), ('F', 'SA'), ('F', 'ZN'), ('FE', 'HD'), ('FE', 'N'), ('FE', 'OA'), ('HD', 'HD'), ('HD', 'I'), ('HD', 'MG'), ('HD', 'MN'), ('HD', 'N'), ('HD', 'NA'), ('HD', 'OA'), ('HD', 'P'), ('HD', 'S'), ('HD', 'SA'), ('HD', 'ZN'), ('MG', 'NA'), ('MG', 'OA'), ('MN', 'N'), ('MN', 'OA'), ('N', 'N'), ('N', 'NA'), ('N', 'OA'), ('N', 'SA'), ('N', 'ZN'), ('NA', 'OA'), ('NA', 'SA'), ('NA', 'ZN'), ('OA', 'OA'), ('OA', 'SA'), ('OA', 'ZN'), ('S', 'ZN'), ('SA', 'ZN')]
        cc_25_rec_types, cc_25_lig_types = zip(*cc_25_types)
        self.cc_25 = close_contacts(protein, cutoff=2.5, protein_types=cc_25_rec_types, ligand_types=cc_25_lig_types, mode='atom_types_ad4', aligned_pairs=True)
        
    def build(self, ligands, protein = None):
        if protein is None:
            protein = self.protein
        protein_dict = protein.atom_dict
        desc = None
        for mol in ligands:
            mol_dict = mol.atom_dict
            vec = np.array([], dtype=float)
            vec = tuple()
            # Vina
            ### TODO: Asynchronous output from vina, push command to score and retrieve at the end?
            scored_mol = self.vina.score(mol, single=True)[0].data
            vina_scores = ['vina_affinity', 'vina_gauss1', 'vina_gauss2', 'vina_repulsion', 'vina_hydrophobic', 'vina_hydrogen']
            vec += tuple([scored_mol[key] for key in vina_scores])
            
            # Close Contacts (<4A)
            vec += tuple(self.cc_4.build(mol, single=True).flatten())
            
            # Electrostatics (<4A)
            ele_types = (('A', 'A'), ('A', 'C'), ('A', 'CL'), ('A', 'F'), ('A', 'FE'), ('A', 'HD'), ('A', 'MG'), ('A', 'MN'), ('A', 'N'), ('A', 'NA'), ('A', 'OA'), ('A', 'SA'), ('A', 'ZN'), ('BR', 'C'), ('BR', 'HD'), ('BR', 'OA'), ('C', 'C'), ('C', 'CL'), ('C', 'F'), ('C', 'HD'), ('C', 'MG'), ('C', 'MN'), ('C', 'N'), ('C', 'NA'), ('C', 'OA'), ('C', 'SA'), ('C', 'ZN'), ('CL', 'FE'), ('CL', 'HD'), ('CL', 'MG'), ('CL', 'N'), ('CL', 'OA'), ('CL', 'ZN'), ('F', 'HD'), ('F', 'N'), ('F', 'OA'), ('F', 'SA'), ('F', 'ZN'), ('FE', 'HD'), ('FE', 'N'), ('FE', 'OA'), ('HD', 'HD'), ('HD', 'I'), ('HD', 'MG'), ('HD', 'MN'), ('HD', 'N'), ('HD', 'NA'), ('HD', 'OA'), ('HD', 'P'), ('HD', 'S'), ('HD', 'SA'), ('HD', 'ZN'), ('MG', 'NA'), ('MG', 'OA'), ('MN', 'N'), ('MN', 'OA'), ('N', 'N'), ('N', 'NA'), ('N', 'OA'), ('N', 'SA'), ('N', 'ZN'), ('NA', 'OA'), ('NA', 'SA'), ('NA', 'ZN'), ('OA', 'OA'), ('OA', 'SA'), ('OA', 'ZN'), ('S', 'ZN'), ('SA', 'ZN'), ('A', 'BR'), ('A', 'I'), ('A', 'P'), ('A', 'S'), ('BR', 'N'), ('BR', 'SA'), ('C', 'FE'), ('C', 'I'), ('C', 'P'), ('C', 'S'), ('CL', 'MN'), ('CL', 'NA'), ('CL', 'P'), ('CL', 'S'), ('CL', 'SA'), ('CU', 'HD'), ('CU', 'N'), ('FE', 'NA'), ('FE', 'SA'), ('I', 'N'), ('I', 'OA'), ('MG', 'N'), ('MG', 'P'), ('MG', 'S'), ('MG', 'SA'), ('MN', 'NA'), ('MN', 'P'), ('MN', 'S'), ('MN', 'SA'), ('N', 'P'), ('N', 'S'), ('NA', 'P'), ('NA', 'S'), ('OA', 'P'), ('OA', 'S'), ('P', 'S'), ('P', 'SA'), ('P', 'ZN'), ('S', 'SA'), ('SA', 'SA'))
            ele_rec_types, ele_lig_types = zip(*ele_types)
            ele_mol_atoms = atoms_by_type(mol_dict, ele_lig_types, 'atom_types_ad4')
            ele_rec_atoms = atoms_by_type(protein_dict, ele_rec_types, 'atom_types_ad4')
            ele = tuple()
            for r_t, m_t in ele_types:
                mol_ele_dict, rec_ele_dict = interactions.close_contacts(ele_mol_atoms[m_t], ele_rec_atoms[r_t], 4)
                if len(mol_ele_dict) and len(rec_ele_dict):
                    ele += (mol_ele_dict['charge'] * rec_ele_dict['charge']/ np.sqrt((mol_ele_dict['coords'] - rec_ele_dict['coords'])**2).sum(axis=-1) * 138.94238460104697e4).sum(), # convert to J/mol
                else:
                    ele += 0,
            vec += tuple(ele)
            
            # Ligand Atom Types
            ligand_atom_types = ['A', 'BR', 'C', 'CL', 'F', 'HD', 'I', 'N', 'NA', 'OA', 'P', 'S', 'SA']
            atoms = atoms_by_type(mol_dict, ligand_atom_types, 'atom_types_ad4')
            atoms_counts = [len(atoms[t]) for t in ligand_atom_types]
            vec += tuple(atoms_counts)
            
            # Close Contacts (<2.5A)
            vec += tuple(self.cc_25.build(mol, single=True).flatten())
            
            # H-Bonds (<4A)
            hbond_mol, hbond_rec, strict = interactions.hbond(mol, protein, 4)
            # Retain only strict hbonds
            hbond_mol = hbond_mol[strict]
            hbond_rec = hbond_rec[strict]
            backbone = hbond_rec['isbackbone']
            alpha = hbond_rec['isalpha']
            beta = hbond_rec['isbeta']
            other = ~alpha & ~beta
            donor_mol = hbond_mol['isdonor']
            donor_rec = hbond_rec['isdonor']
            hbond_vec = ((donor_mol & backbone & alpha).sum(), (donor_mol & backbone & beta).sum(), (donor_mol & backbone & other).sum(),
                        (donor_mol & ~backbone & alpha).sum(), (donor_mol & ~backbone & beta).sum(), (donor_mol & ~backbone & other).sum(),
                        (donor_rec & backbone & alpha).sum(), (donor_rec & backbone & beta).sum(), (donor_rec & backbone & other).sum(),
                        (donor_rec & ~backbone & alpha).sum(), (donor_rec & ~backbone & beta).sum(), (donor_rec & ~backbone & other).sum())
            vec += tuple(hbond_vec)
            
            # Hydrophobic contacts (<4A)
            hydrophobic = interactions.hydrophobic_contacts(mol, protein, 4)[1]
            backbone = hydrophobic['isbackbone']
            alpha = hydrophobic['isalpha']
            beta = hydrophobic['isbeta']
            other = ~alpha & ~beta
            hyd_vec = ((backbone & alpha).sum(), (backbone & beta).sum(), (backbone & other).sum(),
                       (~backbone & alpha).sum(), (~backbone & beta).sum(), (~backbone & other).sum(), len(hydrophobic))
            vec += tuple(hyd_vec)
            
            # Pi-stacking (<7.5A)
            pi_mol, pi_rec, pi_paralel, pi_tshaped = interactions.pi_stacking(mol, protein, 7.5)
            alpha = pi_rec['isalpha'] & pi_paralel
            beta = pi_rec['isbeta'] & pi_paralel
            other = ~alpha & ~beta & pi_paralel
            pi_vec = (alpha.sum(), beta.sum(), other.sum())
            vec += tuple(pi_vec)
            
            # count T-shaped Pi-Pi interaction
            alpha = pi_rec['isalpha'] & pi_tshaped
            beta = pi_rec['isbeta'] & pi_tshaped
            other = ~alpha & ~beta & pi_tshaped
            pi_t_vec = (alpha.sum(), beta.sum(), other.sum())
            
            # Pi-cation (<6A)
            pi_rec, cat_mol, strict = interactions.pi_cation(protein, mol, 6)
            alpha = pi_rec['isalpha'] & strict
            beta = pi_rec['isbeta'] & strict
            other = ~alpha & ~beta & strict
            pi_cat_vec = (alpha.sum(), beta.sum(), other.sum())
            
            pi_mol, cat_rec, strict = interactions.pi_cation(mol, protein, 6)
            alpha = cat_rec['isalpha'] & strict
            beta = cat_rec['isbeta'] & strict
            other = ~alpha & ~beta & strict
            pi_cat_vec += (alpha.sum(), beta.sum(), other.sum())
            
            vec += tuple(pi_cat_vec)
            
            # T-shape (perpendicular Pi's) (<7.5A)
            vec += tuple(pi_t_vec)
            
            # Active site flexibility (<4A)
            acitve_site = interactions.close_contacts(mol_dict, protein_dict, 4)[1]
            backbone = acitve_site['isbackbone']
            alpha = acitve_site['isalpha']
            beta = acitve_site['isbeta']
            other = ~alpha & ~beta
            as_flex = ((backbone & alpha).sum(), (backbone & beta).sum(), (backbone & other).sum(),
                       (~backbone & alpha).sum(), (~backbone & beta).sum(), (~backbone & other).sum(), len(acitve_site))
            vec += tuple(as_flex)
            
            # Salt bridges (<5.5)
            salt_bridges = interactions.salt_bridges(mol, protein, 5.5)[1]
            vec += (salt_bridges['isalpha'].sum(), salt_bridges['isbeta'].sum(),
                                   (~salt_bridges['isalpha'] & ~salt_bridges['isbeta']).sum(), len(salt_bridges))
            
            # Rotatable bonds
            vec += mol.num_rotors,
            
            if desc is None:
                desc = np.zeros(len(vec), dtype=float)
            desc = np.vstack((desc, np.array(vec, dtype=float)))
        
        return desc[1:]
