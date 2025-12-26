import io
import base64
from nicegui import ui
from chempleter.inference import handle_prompt, extend
from pathlib import Path
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles
from importlib import resources


def build_chempleter_ui():
    """
    Build Chempleter GUI using Nicegui. This fucntion also reads in the trained model, vocabulary files.
    """

    def _validate_smiles(smiles):
        try:
            prompt = handle_prompt(
                smiles,
                selfies=None,
                stoi=None,
                alter_prompt=alter_prompt_checkbox.value,
            )
            return True, prompt
        except Exception as e:
            print(e)
            return False, ""

    def show_generated_molecule():
        if _validate_smiles(smiles=smiles_input.value)[0] is False:
            ui.notify("Error parsing input SMILES", type="negative")
            return

        smiles_input.disable()
        generate_button.set_text("Generating...")

        # set parameters from gui
        length = length_slider.value
        max_len = int(length * 100)
        min_len = int((length / 2) * 100)
        next_atom_criteria = (
            "greedy" if sampling_radio.value == "Most probable" else "top_k_temperature"
        )

        # generate
        generated_molecule, generated_smiles, _ = extend(
            model=None,
            stoi_file=None,
            itos_file=None,
            smiles=smiles_input.value,
            min_len=min_len,
            max_len=max_len,
            temperature=temperature_input.value,
            alter_prompt=alter_prompt_checkbox.value,
            next_atom_criteria=next_atom_criteria,
        )

        # check if same
        if generated_smiles == smiles_input.value:
            if alter_prompt_checkbox.value is False:
                ui.notify(
                    "Same molecule as input. Try allowing prompt modification.",
                    type="negative",
                )

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
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        molecule_image.set_source(f"data:image/png;base64,{img_base64}")

        # after image generation
        smiles_input.enable()
        generate_button.set_text("Generate")

    logo_path = Path(resources.files("chempleter.data").joinpath("chempleter_logo.png"))
    ui.page_title("Chempleter")
    with ui.column().classes("w-full min-h-screen items-center overflow-auto py-8"):
        with ui.row(wrap=False).classes("w-128 justify-center"):
            ui.image(logo_path).classes("w-64")

        with ui.card().tight():
            with ui.row(wrap=False).classes("w-128 justify-center"):
                smiles_input = ui.input(
                    "Enter SMILES",
                    validation=lambda value: "Invalid SMILES"
                    if _validate_smiles(value)[0] is False
                    else None,
                )

            with ui.row(wrap=True).classes("w-full justify-center items-center"):
                ui.separator()
                alter_prompt_checkbox = ui.checkbox("Allow prompt modification")
                temperature_input = ui.number(
                    "Temperature", precision=1, value=0.7, step=0.1, min=0.2, max=5
                ).classes("w-32")
                ui.separator()
            with ui.row(wrap=False):
                ui.chip("Sampling: ", color="white")
                sampling_radio = ui.radio(
                    ["Most probable", "Random"], value="Random"
                ).props("inline")
            with ui.row(wrap=False).classes("w-128"):
                ui.chip("Molecule size: ", color="white")
                ui.chip("Smaller")
                length_slider = ui.slider(min=0.1, max=1, step=0.05, value=0.5)
                ui.chip("Larger")

        with ui.row():
            ui.label("Processed prompt :")
            ui.label().bind_text_from(smiles_input, "value", backward=_validate_smiles)

        with ui.row().classes("w-128 justify-center"):
            generate_button = ui.button("Generate", on_click=show_generated_molecule)

        with ui.card(align_items="center").tight().classes("w-128 justify-center"):
            molecule_image = ui.image().style("width: 300px")
            with ui.card_section():
                generated_smiles_label = ui.label("")


def run_chempleter_gui():
    """
    This function runs the ui.run and acts as the entry point for the script chempleter-gui
    """
    favicon_path = Path(resources.files("chempleter.data").joinpath("chempleter.ico"))
    ui.run(favicon=favicon_path, reload=False, root=build_chempleter_ui)


if __name__ in {"__main__", "__mp_main__"}:
    run_chempleter_gui()
