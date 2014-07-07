from tempfile import mkdtemp
from shutil import rmtree
from os.path import exists
import subprocess
import numpy as np
import re

class autodock_vina:
    def __init__(self, protein, size=(10,10,10), center=(0,0,0), auto_ligand=None, exhaustivness=8, num_modes=9, energy_range=3, seed=None, prefix_dir='/tmp', ncpu=1, executable=None, autocleanup=True):
        self.dir = mkdtemp(dir = prefix_dir, prefix='autodock_vina_')
        # define binding site
        self.size = size
        self.center = center
        # center automaticaly on ligand
        if auto_ligand:
            self.center = tuple(np.array([atom.coords for atom in ligand], dtype=np.float16).mean(axis=0))
        # autodetect Vina executable
        if not executable:
            self.executable = subprocess.check_output(['which', 'vina']).split('\n')[0]
        else:
            self.executable = executable
        # detect version
        self.version = subprocess.check_output([self.executable, '--version']).split(' ')[2]
        self.autocleanup = autocleanup
        
        
        # write protein to file
        self.protein_file = self.dir + '/protein.pdbqt'
        protein.write('pdbqt', self.protein_file, opt={'r':None,})
        
        #pregenerate common Vina parameters
        self.params = []
        self.params = self.params + ['--center_x', str(self.center[0]), '--center_y', str(self.center[1]), '--center_z', str(self.center[2])]
        self.params = self.params + ['--size_x', str(self.size[0]), '--size_y', str(self.size[1]), '--size_z', str(self.size[2])]
        self.params = self.params + ['--cpu', str(ncpu)]
        self.params = self.params + ['--exhaustiveness', str(exhaustivness)]
        if not seed is None:
            self.params = self.params + ['--seed', str(seed)]
        self.params = self.params + ['--num_modes', str(num_modes)]
        self.params = self.params + ['--energy_range', str(energy_range)]
    
    def score(self, ligands):
        output_array = []
        n = 1
        ligand_dir = mkdtemp(dir = self.dir, prefix='ligands_')
        for ligand in ligands:
            # write ligand to file
            ligand_file = ligand_dir + '/' + str(n) + '.pdbqt'
            ligand.write('pdbqt', ligand_file, overwrite=True)
            output_array.append(parse_vina_scoring_output(subprocess.check_output([self.executable, '--score_only', '--receptor', self.protein_file, '--ligand', ligand_file] + self.params)))
            n +=1
        return output_array
            
    def dock(self, ligands):
        output_array = []
        n = 1
        ligand_dir = mkdtemp(dir = self.dir, prefix='ligands_')
        for ligand in ligands:
            # write ligand to file
            ligand_file = ligand_dir + '/' + str(n) + '.pdbqt'
            ligand_outfile = ligand_dir + '/' + str(n) + '_out.pdbqt'
            ligand.write('pdbqt', ligand_file, overwrite=True)
            output_array.append(parse_vina_docking_output(subprocess.check_output([self.executable, '--receptor', self.protein_file, '--ligand', ligand_file, '--out', ligand_outfile] + self.params)))
            n +=1
        return output_array
    
    
    # !!! FIX: this does not delete directory for some reason
    def __enter__(self):
        return self
    
    def __exit__(self):
        if exists(self.dir) and self.autocleanup:
            rmtree(self.dir)
            dasdsad
    
def parse_vina_scoring_output(output):
    out = {}
    r = re.compile('^(Affinity:|\s{4})')
    for line in output.split('\n')[13:]: # skip some output
        if r.match(line):
            m = line.replace(' ','').split(':')
            if m[0] == 'Affinity':
                m[1] = m[1].replace('(kcal/mol)','')
            out[m[0].lower()] = float(m[1])
    return out
    
def parse_vina_docking_output(output):
    out = []
    r = re.compile('^\s+\d\s+')
    for line in output.split('\n')[13:]: # skip some output
        if r.match(line):
            s = line.split()
            out.append({'affinity': s[1], 'rmsd_lb': s[2], 'rmsd_ub': s[3]})
    return out
    
