from multiprocessing import Pool

import toolkits.ob as toolkit



class virtualscreening:
    def __init__(self, cpus=-1, verbose=False):
        self._pipe = None
        self.cpus = cpus
        self.num_input = 0
        self.num_output = 0
        self.verbose = verbose
        
    def load_ligands(self, file_type, ligands_file):
        self._pipe = self._ligand_pipe(toolkit.readfile(file_type, ligands_file))
    
    def _ligand_pipe(self, ligands):
        for n, mol in enumerate(ligands):
            self.num_input = n+1 
            yield mol
    
    def apply_filter(self, expression, filter_type='expression', soft_fail = 0):
        if filter_type == 'expression':
            self._pipe = self._filter(expression, soft_fail = soft_fail)
        elif filter_type == 'preset':
            # define presets
            # TODO: move presets to another config file
            # Lipinski rule of 5's
            if expression.lower() in ['l5', 'ro5']:
                self._pipe = self._filter(self._pipe, ['mol.molwt < 500', 'mol.calcdesc(["HBA1"])["HBA1"] <= 10', 'mol.calcdesc(["HBD"])["HBD"] <= 5', 'mol.calcdesc(["logP"])["logP"] <= 5'], soft_fail = soft_fail)
            # Rule of three
            elif expression.lower() in ['ro3']:
                self._pipe = self._filter(self._pipe, ['mol.molwt < 300', 'mol.calcdesc(["HBA1"])["HBA1"] <= 3', 'mol.calcdesc(["HBD"])["HBD"] <= 3', 'mol.calcdesc(["logP"])["logP"] <= 3'], soft_fail = soft_fail)
    
    def _filter(self, pipe, expression, soft_fail = 0):
        for mol in pipe:
            if type(expression) is list:
                fail = 0
                for e in expression:
                    if not eval(e):
                        fail += 1
                if fail <= soft_fail:
                    yield mol
            else:
                if eval(expression):
                    yield mol
    
#    def score(self):
#    
    def write(self, *args, **kwargs):
        output_file = toolkit.Outputfile(*args, **kwargs)
        for mol in self.fetch():
            output_file.write(mol)
        output_file.close()
    
#    def write_csv():
    def fetch(self):
        for n, mol in enumerate(self._pipe):
            self.num_output = n+1 
            if self.verbose and self.num_input % 100 == 0:
                print "\rPassed: %i (%.2f%%)\tTotal: %i" % (self.num_output, float(self.num_output)/float(self.num_input)*100, self.num_input),
            yield mol
        if self.verbose:
            print ""
