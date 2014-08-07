from pybel import *
import copy_reg

### Monkeypatch pybel objects pickling
pickle_format = 'mol2'
def pickle_mol(self):
    return unpickle_mol, (self.write(pickle_format), dict(self.data.items()))

def unpickle_mol(string, data):
    mol = readstring(pickle_format, string)
    mol.data.update(data)
    return mol

copy_reg.pickle(Molecule, pickle_mol, unpickle_mol)