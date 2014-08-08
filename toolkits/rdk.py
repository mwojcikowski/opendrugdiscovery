import numpy as np
from cinfony import rdk
from cinfony.rdk import *

class Fingerprint(rdk.Fingerprint):
    @property
    def raw(self):
        return np.array(self.fp, dtype=bool)
        
rdk.Fingerprint = Fingerprint
