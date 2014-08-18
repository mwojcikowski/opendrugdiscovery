import csv
import numpy as np
from multiprocessing import Pool

from oddt.toolkits import ob as toolkit
from oddt.scoring import scorer
from oddt.scoring.machinelearning.randomforest import randomforest
from oddt.scoring.descriptors import close_contacts

# RF-Score settings
ligand_atomic_nums = [6,7,8,9,15,16,17,35,53]
protein_atomic_nums = [6,7,8,16]
cutoff = 12

# define sub-function for paralelization
def generate_descriptor(packed):
    pdbid, gen, pdbbind_dir, pdbbind_version = packed
    protein_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_protein.pdb" % (pdbid, pdbid)
    ligand_file = pdbbind_dir + "/v" + pdbbind_version + "/%s/%s_ligand.sdf" % (pdbid, pdbid)
    protein = toolkit.readfile("pdb", protein_file, opt = {'b': None}).next()
    ligand = toolkit.readfile("sdf", ligand_file).next()
    return gen.build([ligand], protein).flatten()

 # skip comments and merge multiple spaces
def _csv_file_filter(f):
    for row in open(f, 'rb'):
        if row[0] == '#':
            continue
        yield ' '.join(row.split())

class rfscore(scorer):
    def __init__(self, protein = None, n_jobs = -1, **kwargs):
        self.protein = protein
        self.n_jobs = n_jobs
        model = randomforest(n_estimators = 500, oob_score = True, n_jobs = n_jobs, **kwargs)
        descriptors = close_contacts(protein, cutoff = cutoff, protein_types = protein_atomic_nums, ligand_types = ligand_atomic_nums)
        super(rfscore,self).__init__(model, descriptors)
    
    def train(self, pdbbind_dir, pdbbind_version = '2013', sf_pickle = ''):
        core_desc = np.zeros((1,36), dtype=int)
        core_act = np.zeros(1, dtype=float)
        refined_desc = np.zeros((1,36), dtype=int)
        refined_act = np.zeros(1, dtype=float)
        
        # build train and test 
        cpus = self.n_jobs if self.n_jobs > 0 else None
        pool = Pool(processes=cpus)
        
        core_set = []
        for row in csv.reader(_csv_file_filter(pdbbind_dir + "/v" + pdbbind_version + "/INDEX_core_data." + pdbbind_version), delimiter=' '):
            pdbid = row[0]
            act = float(row[3])
            core_set.append(pdbid)
            core_act = np.vstack((core_act, act))
        
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in core_set])
        core_desc = np.vstack(result)
        core_act = core_act[1:]
        
        refined_set = []
        for row in csv.reader(_csv_file_filter(pdbbind_dir + "/v" + pdbbind_version + "/INDEX_refined_data." + pdbbind_version), delimiter=' '):
            pdbid = row[0]
            act = float(row[3])
            if pdbid in core_set:
                continue
            refined_set.append(pdbid)
            refined_act = np.vstack((refined_act, act))
        
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in refined_set])
        refined_desc = np.vstack(result)
        refined_act = refined_act[1:]
        
        np.savetxt('RF-Score_descriptors.csv', np.hstack((core_act, core_desc)), delimiter=',', fmt='%s')
        print refined_act.dtype, refined_desc.dtype
        self.model.fit(refined_desc, refined_act.flatten())
        rmse = np.sqrt(np.mean(np.square(self.model.oob_prediction_ - refined_act.flatten())))
        print 'Train set:', self.model.score(core_desc, core_act.flatten()), 'RMSE:', rmse
        print 'Test set:', self.model.score(refined_desc, refined_act.flatten())
        if sf_pickle:
            self.save(sf_pickle)

