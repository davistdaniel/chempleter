from chempleter.inference import handle_prompt
from rdkit import Chem
from rdkit.Chem import Draw

def _validate_smiles(smiles, alter_prompt_checkbox_value, frag1_smiles=None, frag2_smiles=None,):
    try:
        prompt = handle_prompt(
            smiles,
            selfies=None,
            stoi=None,
            alter_prompt=alter_prompt_checkbox_value,
            frag1_smiles=frag1_smiles,
            frag2_smiles=frag2_smiles,
        )
        return True, prompt
    except Exception as e:
        print(e)
        return False, ""
    
def _draw_if_bridge(frag1_smiles,frag2_smiles,generated_molecule):

    frag1_structure = Chem.MolFromSmiles(frag1_smiles)
    frag2_structure = Chem.MolFromSmiles(frag2_smiles)

    if frag1_structure is None or frag2_structure is None:
        return Draw.MolToImage(generated_molecule, size=(1000, 1000))

    frag_atoms = set()

    matches1 = generated_molecule.GetSubstructMatches(frag1_structure)
    matches2 = generated_molecule.GetSubstructMatches(frag2_structure)

    for match in matches1 + matches2:
        for atom_idx in match:
            frag_atoms.add(atom_idx)

    if not frag_atoms:
        return Draw.MolToImage(generated_molecule, size=(1000, 1000))

    bridge_atoms = [
        atom.GetIdx()
        for atom in generated_molecule.GetAtoms()
        if atom.GetIdx() not in frag_atoms
    ]

    bridge_bonds = []
    for bond in generated_molecule.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 not in frag_atoms and a2 not in frag_atoms:
            bridge_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(
        generated_molecule,
        size=(1000, 1000),
        highlightAtoms=bridge_atoms,
        highlightBonds=bridge_bonds,
    )

    return img

def _draw_if_extend(input_smiles,generated_molecule):
    # try highlighting input molecule
    input_molecule_structure = Chem.MolFromSmiles(input_smiles)
    if input_molecule_structure is None:
        return Draw.MolToImage(generated_molecule, size=(1000, 1000))

    match = generated_molecule.GetSubstructMatch(input_molecule_structure)
    if match is None:
        return Draw.MolToImage(generated_molecule, size=(1000, 1000))

    highlight_atoms = list(match) if match else []
    highlight_bonds = []
    for bond in generated_molecule.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in highlight_atoms and a2 in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(
        generated_molecule,
        size=(1000, 1000),
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
    )

    return img

def _draw_if_evolve(generated_molecule):
    m_gen_list = generated_molecule

    highlight_atoms = []
    highlight_bonds = []

    for i, mol in enumerate(m_gen_list):
        if i == 0:
            highlight_atoms.append([])
            highlight_bonds.append([])
            continue

        prev = m_gen_list[i - 1]

        mcs = Chem.rdFMCS.FindMCS(
            [prev, mol], ringMatchesRingOnly=True, completeRingsOnly=True
        )

        if mcs.smartsString:
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            match = mol.GetSubstructMatch(mcs_mol)

            common_atoms = set(match)
            common_bonds = set(
                mol.GetBondBetweenAtoms(
                    match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]
                ).GetIdx()
                for b in mcs_mol.GetBonds()
            )

            new_atoms = [
                a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in common_atoms
            ]

            new_bonds = [
                b.GetIdx() for b in mol.GetBonds() if b.GetIdx() not in common_bonds
            ]

            highlight_atoms.append(new_atoms)
            highlight_bonds.append(new_bonds)
        else:
            highlight_atoms.append([])
            highlight_bonds.append([])

    img = Draw.MolsToGridImage(
        m_gen_list,
        molsPerRow=min(len(m_gen_list), 4),
        subImgSize=(1000, 1000),
        highlightAtomLists=highlight_atoms,
        highlightBondLists=highlight_bonds,
    )

    return img

def _draw_input_if_decorate(input_smiles):
    initial_mol = Chem.MolFromSmiles(input_smiles)
    for atom in initial_mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    _w, _h = 500 if len(input_smiles)<50 else 1000, 500 if len(input_smiles)<50 else 1000
    img = Draw.MolToImage(initial_mol, size=(_w, _h))

    return img, initial_mol