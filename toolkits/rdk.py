import numpy as np
from cinfony import rdk
from cinfony.rdk import *

class Fingerprint(rdk.Fingerprint):
    @property
    def raw(self):
        return np.array(self.fp, dtype=bool)
        
rdk.Fingerprint = Fingerprint

# Patch reader not to return None as molecules
def _readfile(format, filename):
    for mol in rdk.readfile(format, filename):
        if mol is not None:
            yield mol

rdk.readfile = _readfile
