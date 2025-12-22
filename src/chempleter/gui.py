

import io
import base64
import json
import torch
from nicegui import ui
from chempleter.inference import handle_prompt,extend
from chempleter.model import ChempleterModel
from pathlib import Path
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles
from importlib import resources

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

stoi_file = Path(resources.files("chempleter.data").joinpath("stoi.json"))
itos_file = Path(resources.files("chempleter.data").joinpath("itos.json"))
checkpoint_file = Path(resources.files("chempleter.data").joinpath("model.pt"))

with open(stoi_file) as f:
    stoi = json.load(f)
with open(itos_file) as f:
    itos = json.load(f)

model = ChempleterModel(vocab_size=len(stoi))
checkpoint = torch.load(checkpoint_file,map_location=device,weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])


def _validate_smiles(smiles):
    try:
        prompt = handle_prompt(smiles,selfies=None,stoi=stoi,alter_prompt=alter_prompt_checkbox.value)
        return True,prompt
    except Exception as e:
        print(e)
        return False,""

def show_generated_molecule():
    
    if _validate_smiles(smiles=smiles_input.value)[0] is False:
        ui.notify("Error parsing input SMILES", type='negative')
        return

    smiles_input.disable()
    generate_button.set_text("Generating...")
    
    # set parameters from gui
    length = length_slider.value
    max_len=int(length*100)
    min_len = int((length/4)*100)
    next_atom_criteria = "greedy" if sampling_radio.value == "Most probable" else "top_k_temperature"

    # generate
    generated_molecule,generated_smiles,generated_selfies = extend(model=model,stoi_file=stoi_file,itos_file=itos_file,smiles=smiles_input.value,min_len=min_len,max_len=max_len,alter_prompt=alter_prompt_checkbox.value,next_atom_criteria=next_atom_criteria)

    # check if same
    if generated_smiles == smiles_input.value:
        if alter_prompt_checkbox.value is False:
            ui.notify("Same molecule as input. Try allowing prompt modification.", type='negative')

    # try highlighting input molecule
    input_molecule_structure = MolFromSmiles(smiles_input.value)
    if input_molecule_structure is not None:
        match = generated_molecule.GetSubstructMatch(input_molecule_structure)
        highlight_atoms = list(match) if match else []
        highlight_bonds = []
        for bond in generated_molecule.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in highlight_atoms and a2 in highlight_atoms:
                highlight_bonds.append(bond.GetIdx())
        
        img = Draw.MolToImage(
            generated_molecule,
            size=(300, 300),
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors={i: (1.0, 0.0, 0.0) for i in highlight_atoms},
            highlightBondColors={i: (1.0, 0.0, 0.0) for i in highlight_bonds},
        )
    
    else:
        img = Draw.MolToImage(generated_molecule, size=(300, 300))

    # generated image
    generated_smiles_label.set_text(generated_smiles)
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    molecule_image.set_source(
            f'data:image/png;base64,{img_base64}'
        )
    
    # after image generation
    smiles_input.enable()
    generate_button.set_text("Generate")

with ui.row(wrap=False).classes("w-128"):
    smiles_input = ui.input("Enter SMILES",validation=lambda value: 'Invalid SMILES' if _validate_smiles(value)[0] is False else None)

with ui.card().tight():
    with ui.row(wrap=False):
        alter_prompt_checkbox = ui.checkbox("Allow prompt modification")
    with ui.row(wrap=False):
        ui.chip("Sampling: ",color="white")
        sampling_radio = ui.radio(["Most probable", "Random"], value="Random").props('inline')
    with ui.row(wrap=False).classes("w-128"):
        ui.chip("Molecule size: ",color="white")
        ui.chip("Smaller")
        length_slider = ui.slider(min=0.1, max=1, step=0.05, value=0.5)
        ui.chip("Larger")

with ui.row():
    current_prompt_label = ui.label("Processed prompt :")
    current_prompt = ui.label().bind_text_from(smiles_input, "value",backward=_validate_smiles)

with ui.row():
    generated_smiles_label = ui.label("Generated SMILES:")
    current_generated_smiles = ui.label()


generate_button = ui.button("Generate",on_click=show_generated_molecule)

with ui.card(align_items="center").tight():

    molecule_image  = ui.image().style('width: 300px')
    with ui.card_section():
        generated_smiles_label = ui.label("")

ui.run()