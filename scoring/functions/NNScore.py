import csv
from os.path import dirname, isfile
import numpy as np
from multiprocessing import Pool
import warnings

from oddt import toolkit
from oddt.scoring import scorer
from oddt.scoring.descriptors.binana import binana_descriptor
from oddt.scoring.models.regressors import neuralnetwork

# numpy after pickling gives Runtime Warnings
warnings.simplefilter("ignore", RuntimeWarning)

# define sub-function for paralelization
def generate_descriptor(packed):
    pdbid, gen, pdbbind_dir, pdbbind_version = packed
    protein_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_pocket.pdb" % (pdbid, pdbid)
    if not isfile(protein_file):
        protein_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_protein.pdb" % (pdbid, pdbid)
    ligand_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_ligand.sdf" % (pdbid, pdbid)
    protein = toolkit.readfile("pdb", protein_file).next()
    # mark it as a protein
    protein.protein = True
    ligand = toolkit.readfile("sdf", ligand_file).next()
    return gen.build([ligand], protein).flatten()

# skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        if row[0] == '#':
            continue
        yield ' '.join(row.split())

class nnscore(scorer):
    def __init__(self, protein = None, n_jobs = -1, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        model = []
        decsriptors = binana_descriptor(protein)
        super(nnscore,self).__init__(model, decsriptors, score_title='nnscore')
    
    def train(self, pdbbind_dir, pdbbind_version = '2007', sf_pickle = ''):
        # build train and test 
        cpus = self.n_jobs if self.n_jobs > 0 else None
        pool = Pool(processes=cpus)
        
        core_act = np.zeros(1, dtype=float)
        core_set = []
        if pdbbind_version == '2007':
            csv_file = pdbbind_dir + "/v" + pdbbind_version + "/INDEX." + pdbbind_version + ".core.data"
        else:
            csv_file = pdbbind_dir + "/v" + pdbbind_version + "/INDEX_core_data." + pdbbind_version
        for row in csv.reader(_csv_file_filter(csv_file), delimiter=' '):
            pdbid = row[0]
            act = float(row[3])
            core_set.append(pdbid)
            core_act = np.vstack((core_act, act))
        
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in core_set])
        core_desc = np.vstack(result)
        core_act = core_act[1:]
        
        refined_act = np.zeros(1, dtype=float)
        refined_set = []
        if pdbbind_version == '2007':
            csv_file = pdbbind_dir + "/v" + pdbbind_version + "/INDEX." + pdbbind_version + ".refined.data"
        else:
            csv_file = pdbbind_dir + "/v" + pdbbind_version + "/INDEX_refined_data." + pdbbind_version
        for row in csv.reader(_csv_file_filter(csv_file), delimiter=' '):
            pdbid = row[0]
            act = float(row[3])
            if pdbid in core_set:
                continue
            refined_set.append(pdbid)
            refined_act = np.vstack((refined_act, act))
        
        #result = [generate_descriptor((pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version)) for pdbid in refined_set]
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in refined_set])
        refined_desc = np.vstack(result)
        refined_act = refined_act[1:]
        
        self.train_descs = refined_desc
        self.train_target = refined_act
        
        self.test_descs = core_desc
        self.test_target = core_act
        
        if sf_pickle:
            self.save(sf_pickle)
        else:
            self.save(dirname(__file__) + '/NNscore.pickle')
        
    @classmethod
    def load(self, filename = ''):
        if not filename:
            filename = dirname(__file__) + '/NNscore.pickle'
        return scorer.load(filename)

