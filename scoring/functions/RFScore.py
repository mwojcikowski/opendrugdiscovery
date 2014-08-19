import csv
from os.path import dirname
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

# PDB IDs from original implementation
rfscore_pdbids = ['1hk4', '1ha2', '1gni', '1nhu', '2d3z', '2d3u', '1ajp', '1ai5', '1ajq', '1gpk', '1h23', '1e66', '2rkm', '1b9j', '1b7h', '1u2y', '1u33', '1xd1', '1uwt', '2ceq', '2cer', '2qwb', '2qwd', '2qwe', '2j77', '2j78', '2cet', '1m0n', '1zc9', '1m0q', '2fdp', '1fkn', '2g94', '1v16', '1olu', '1ols', '1tok', '1toj', '1toi', '1n2v', '1k4g', '1s39', '1kv1', '2bak', '2baj', '1ndw', '1ndy', '1ndz', '2hdq', '1l2s', '1xgj', '1q8t', '1ydt', '1re8', '1g7q', '1fzj', '1fzk', '1m2q', '1om1', '1zoe', '2azr', '1g7f', '1nny', '5er1', '2er9', '4er2', '1ppm', '1apw', '1bxo', '1nje', '1tsy', '1nja', '4tln', '1tmn', '4tmn', '1fh7', '1fh9', '1fh8', '2fzc', '2h3e', '1d09', '2ctc', '8cpa', '7cpa', '1e1v', '1b39', '1pxo', '1bcu', '1vzq', '1sl3', '2brb', '2brm', '1nvq', '1jqd', '2aov', '2aou', '1y1m', '1pb9', '1pbq', '1vfn', '1v48', '1b8o', '1p1q', '1syh', '1ftm', '1fcx', '1fd0', '1fcz', '1f4e', '1f4f', '1f4g', '1f5k', '1o3p', '1sqa', '2b1v', '2fai', '2ayr', '1avn', '1ttm', '1if7', '2bok', '1nfy', '1mq6', '2usn', '2d1o', '1hfs', '2flb', '2bz6', '2b7d', '1loq', '1lol', '1x1z', '4tim', '1kv5', '1trd', '1bra', '1j16', '1j17', '1utp', '1v2o', '1o3f', '1jys', '1nc1', '1y6q', '1bma', '1ela', '1elb', '1pr5', '1a69', '1k9s', '3pce', '3pch', '3pcj', '1pz5', '2cgr', '1flr', '2gss', '3gss', '10gs', '6std', '2std', '3std', '1jaq', '1zs0', '1zvx', '2d0k', '1dhi', '2drc', '1slg', '1df8', '2f01', '2g8r', '1o0h', '1u1b', '2c02', '1hi4', '2bzz', '1tyr', '1e5a', '2g5u', '1sv3', '1q7a', '1jq9', '1a08', '1a1b', '1is0', '1a30', '2f80', '2i0d', '1d7j', '1fki', '1fkb', '6rnt', '1det', '1rnt']

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
            if pdbid not in rfscore_pdbids:
                continue
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
        
        result = pool.map(generate_descriptor, [(pdbid, self.descriptor_generator, pdbbind_dir, pdbbind_version) for pdbid in refined_set])
        refined_desc = np.vstack(result)
        refined_act = refined_act[1:]
        
        self.train_descs = refined_desc
        self.train_target = refined_act
        
        self.test_descs = core_desc
        self.test_target = core_act
        
        self.model.fit(refined_desc, refined_act.flatten())
        
        r2 = self.model.score(core_desc, core_act.flatten())
        r = np.sqrt(r2)
        print 'Test set: R**2:', r2, ' R:', r
        
        rmse = np.sqrt(np.mean(np.square(self.model.oob_prediction_ - refined_act.flatten())))
        r2 = self.model.score(refined_desc, refined_act.flatten())
        r = np.sqrt(r2)
        print 'Train set: R**2:', r2, ' R:', r, 'RMSE:', rmse
        
        if sf_pickle:
            self.save(sf_pickle)
        else:
            self.save(dirname(__file__) + '/RFscore.pickle')
        
    @classmethod
    def load(self, filename = ''):
        if not filename:
            filename = dirname(__file__) + '/RFscore.pickle'
        return scorer.load(filename)
