

import io
import base64
import json
import torch
from nicegui import ui
from chempleter.inference import handle_prompt,extend,handle_len
from chempleter.model import ChempleterModel
from chempleter.inference import extend
from pathlib import Path
from rdkit.Chem import Draw
from importlib import resources

stoi_file = Path(resources.files("chempleter.data").joinpath("selfies_to_integer_qm9_zinc.json"))
itos_file = Path(resources.files("chempleter.data").joinpath("integer_to_selfies_qm9_zinc.json"))
checkpoint_file = Path(resources.files("chempleter.data").joinpath("checkpoint.pt"))


with open(stoi_file) as f:
    stoi = json.load(f)
with open(itos_file) as f:
    itos = json.load(f)


model = ChempleterModel(vocab_size=len(stoi))
checkpoint = torch.load(checkpoint_file,weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])


def _validate_smiles(smiles):
    try:
        prompt = handle_prompt(smiles,selfies=None,stoi=stoi,alter_prompt=alter_prompt_checkbox.value)
        return True,prompt
    except Exception as e:
        print(e)
        return False,""

def show_generated_molecule():
    smiles_input.disable()
    prompt=current_prompt.text
    length = length_slider.value
    max_len=int(length*100)
    min_len = int((length/4)*100)
    m,generated_smiles,generated_selfies = extend(model=model,stoi_file=stoi_file,itos_file=itos_file,smiles=smiles_input.value,min_len=min_len,max_len=max_len,alter_prompt=alter_prompt_checkbox.value)

    if generated_smiles == smiles_input.value:
        if alter_prompt_checkbox.value is False:
            ui.notify("Same molecule as input. Try allowing prompt alteration.")

    current_generated_smiles.set_text(generated_smiles)
    img = Draw.MolToImage(m, size=(300, 300))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    molecule_image.set_source(
            f'data:image/png;base64,{img_base64}'
        )
    smiles_input.enable()
    

with ui.row():
    smiles_input = ui.input("Enter SMILES",validation=lambda value: 'Invalid SMILES' if _validate_smiles(value)[0] is False else None)
    alter_prompt_checkbox = ui.checkbox("Alter prompt")

with ui.row():
    current_prompt_label = ui.label("Current Prompt :")
    current_prompt = ui.label().bind_text_from(smiles_input, "value",backward=_validate_smiles)

ui.label("Molecule length")

with ui.row():
    ui.label("shorter")
    length_slider = ui.slider(min=0.1, max=1, step=0.1, value=0.5)
    ui.label("longer")

with ui.row():
    generated_smiles_label = ui.label("Generated :")
    current_generated_smiles = ui.label()


generate_button = ui.button("Generate!",on_click=show_generated_molecule)
molecule_image  = ui.image().style('width: 300px')


ui.run()