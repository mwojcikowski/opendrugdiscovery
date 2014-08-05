from pybel import *
from zlib import compress, decompress
import copy_reg

### Monkeypatch pybel objects pickling
# default pickle format - mol2 as the most comrahensive common one
pickle_format = 'mol2'
def pickle_mol(self):
    return unpickle_mol, (self.write(pickle_format),)

def unpickle_mol(string):
    return readstring(pickle_format, string)
copy_reg.pickle(Molecule, pickle_mol, unpickle_mol)
