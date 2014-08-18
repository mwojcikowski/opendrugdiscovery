from oddt.scoring.descriptors.binana import binana_descriptor
from oddt.scoring.machinelearning.neuralnetwork import neuralnetwork

class nnscore(scorer):
    def __init__(self, protein = None):
        self.protein = protein
        model = neuralnetwork()
        decsriptors = binana_descriptor(protein)
        super(nnscore,self).__init__(model, decsriptors)
    
    def train(pdbbind_dir):
        pass
