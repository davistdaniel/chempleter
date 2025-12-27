from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
try:
    from rdkit.Contrib.SA_Score import sascorer # type: ignore
except ImportError:
    sascorer = None

def calculate_descriptors(m):
    
    # features realed to structure
    fsp3 = rdMolDescriptors.CalcFractionCSP3(m)
    rotatable_bonds = Lipinski.NumRotatableBonds(m)
    rings = rdMolDescriptors.CalcNumRings(m)
    
    # sa score ranging from 1 (easy to make) and 10 (very difficult to make) :  J Cheminform 1, 8 (2009)
    sa_score = sascorer.calculateScore(m) if sascorer else "N/A"
    
    return {
        "MW": round(Descriptors.MolWt(m), 2), # molecular weight
        "LogP": round(Descriptors.MolLogP(m), 2), # partition coeffecient
        "SA_Score": round(sa_score, 2) if isinstance(sa_score, float) else sa_score,
        "QED": round(QED.qed(m), 3), # quantitative estimation of drug-likeness
        "Fsp3": round(fsp3, 2),
        "RotatableBonds": rotatable_bonds,
        "RingCount": rings,
        "TPSA": round(Descriptors.TPSA(m), 2),
        "RadicalElectrons" : Descriptors.NumRadicalElectrons(m)
    }