# tests for chempleter.descriptors

from rdkit import Chem
from chempleter.descriptors import calculate_descriptors

class TestChempleterDescriptors():
    def test_calculate_descriptors(self):
        exp_dict = {'MW': 78.11,
                    'LogP': 1.69,
                    'SA_Score': 1.0,
                    'QED': 0.443,
                    'Fsp3': 0.0,
                    'RotatableBonds': 0,
                    'RingCount': 1,
                    'TPSA': 0.0,
                    'RadicalElectrons': 0}
        
        m = Chem.MolFromSmiles("c1ccccc1")
        test_dict = calculate_descriptors(m)

        assert exp_dict == test_dict