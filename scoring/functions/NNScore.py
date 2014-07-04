import csv
import numpy as np
from machinelearning.neuralnetwork import neuralnetwork
from descriptors import close_contact, Molecule
import pybel

ligand_atomic_num = [6,7,8,9,15,16,17,35,53]
protein_atomic_num = [6,7,8,16]

# 12A cutoff
cutoff = 12

# Supress Pybel warnings
pybel.ob.obErrorLog.StopLogging()

def score(protein_file, ligand_file, protein_file_type='pdb', ligand_file_type='sdf', descriptors_file='RF-Score_descriptors.csv'):
    """ Calculate scoring function using ligands and protein """
    # build forest ####len(ligand_atomic_num)*len(protein_atomic_num)
    nn = neuralnetwork((36, 5, 1))
    
    # train forest on prebuilt descriptors from PDBind 2007
    train_data = np.loadtxt(descriptors_file, delimiter=',')
    nn.train(train_data[:,1:],train_data[:,0], train_alg='bfgs')
    
    # read files
    protein = Molecule(pybel.readfile(protein_file_type, protein_file, opt = {'b': None}).next())
    descs = np.zeros((1,36))
    # build descriptors for ligand
    for lig in pybel.readfile(ligand_file_type, ligand_file):
        ligand = Molecule(lig)
        #print lig.title
        desc = close_contact(protein.coordinate_dict(protein.atom_dict_atomicnum(protein_atomic_num)), ligand.coordinate_dict(ligand.atom_dict_atomicnum(ligand_atomic_num)), cutoff).flatten()
        descs = np.vstack((descs, desc))
    
    # build descriptors and score molecules
    print nn.predict(descs[1:])
    
def prepare():
    """ Prepare base descriptors to train generic RF-Score """
    train_set = np.zeros((1,37)) # 32 descriptors and #33 activity
    for pdbid, act in csv.reader(open('functions/RFScore/PDBbind_core07.txt', 'rb'), delimiter='\t'):
        
        protein_file = "functions/RFScore/v2007/%s/%s_protein.pdb" % (pdbid, pdbid)
        ligand_file = "functions/RFScore/v2007/%s/%s_ligand.sdf" % (pdbid, pdbid)
        
        protein = Molecule(pybel.readfile("pdb", protein_file, opt = {'b': None}).next())
        ligand = Molecule(pybel.readfile("sdf", ligand_file).next())
        
        #print protein.coordinate_dict(ligand_atomic_num)
        #print np.nonzero(protein.coordinate_dict(ligand_atomic_num))
        desc = close_contact(protein.coordinate_dict(protein_atomic_num), ligand.coordinate_dict(ligand_atomic_num), cutoff).flatten()
        train_set = np.vstack((train_set, np.append(act, desc)))
    np.savetxt('RF-Score_descriptors.csv', train_set[1:], delimiter=',', fmt='%s')
    print train_set[1:,1:]
    
#def validate():

