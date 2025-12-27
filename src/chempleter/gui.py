import io
import base64
import json
import torch
from nicegui import ui
from chempleter.inference import handle_prompt, extend
from chempleter.model import ChempleterModel
from chempleter.descriptors import calculate_descriptors
from pathlib import Path
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles
from importlib import resources
from chempleter import __version__


def build_chempleter_ui():
    """
    Build Chempleter GUI using Nicegui. This fucntion also reads in the trained model, vocabulary files.
    """

    # load data
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    default_stoi_file = Path(resources.files("chempleter.data").joinpath("stoi.json"))
    default_itos_file = Path(resources.files("chempleter.data").joinpath("itos.json"))
    default_checkpoint_file = Path(
        resources.files("chempleter.data").joinpath("model.pt")
    )

    with open(default_stoi_file) as f:
        stoi = json.load(f)

    model = ChempleterModel(vocab_size=len(stoi))
    checkpoint = torch.load(
        default_checkpoint_file, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

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
            model=model,
            stoi_file=default_stoi_file,
            itos_file=default_itos_file,
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

        calculated_descriptors = calculate_descriptors(generated_molecule)
        mw_chip.set_text(f"MW: {calculated_descriptors["MW"]}")
        logp_chip.set_text(f"LogP: {calculated_descriptors["LogP"]}")
        SA_score_chip.set_text(f"SA score: {calculated_descriptors["SA_Score"]}")
        qed_chip.set_text(f"QED: {calculated_descriptors["QED"]}")
        fsp3_chip.set_text(f"Fsp3: {calculated_descriptors["Fsp3"]}")
        #tpsa_chip.set_text(f"TPSA: {calculated_descriptors["TPSA"]}")

        # after image generation
        smiles_input.enable()
        generate_button.set_text("Generate")



    logo_path = Path(resources.files("chempleter.data").joinpath("chempleter_logo.png"))
    ui.page_title("Chempleter")
    with ui.column().classes("w-full min-h-screen items-center overflow-auto py-8"):
        with ui.row(wrap=False).classes("w-128 justify-center"):
            with ui.link(target='https://github.com/davistdaniel/chempleter'):
                ui.image(logo_path).classes("w-64")

        with ui.card().tight():
            with ui.row(wrap=False).classes("w-128 justify-center"):
                smiles_input = ui.input(
                    "Enter SMILES",
                    placeholder="c1ccccc1",
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

        with ui.row():
            mw_chip = ui.chip("MW",color='blue-3').tooltip("Molecular weight including hydrogens")
            logp_chip = ui.chip("LogP",color='green-3').tooltip("Octanol-Water Partition Coeffecient")
            SA_score_chip = ui.chip("SA score",color='orange-3').tooltip("Synthetic Accessibility score ranging from 1 (easy) to 10 (difficult)")
            qed_chip = ui.chip("QED",color='grey-3').tooltip("Quantitative Estimate of Drug-likeness ranging from 0 to 1.")
            fsp3_chip = ui.chip("Fsp3",color='pink-3').tooltip("Fraction of sp3 Hybridized Carbons")
            #tpsa_chip = ui.chip("TPSA",color='violet-3').tooltip("Topological polar surface area")
            
        with ui.card(align_items="center").tight().classes("w- 256 justify-center"):
            with ui.card_section():
                generated_smiles_label = ui.label("").style("font-weight: normal; color: black; font-size: 12px;")
            molecule_image = ui.image().style("width: 300px")

    with (
        ui.footer()
        .classes("justify-center")
        .style(
            "height: 30px; text-align: center; padding: 2px; "
            "font-size: 15px; background-color: white; color: grey;"
        )
    ):
        ui.label(f"Chempleter v.{__version__}.")
        ui.link("View on GitHub", "https://github.com/davistdaniel/chempleter").style(
            "font-weight: normal; color: grey; font-size: 15px; "
        )


def run_chempleter_gui():
    """
    This function runs the ui.run and acts as the entry point for the script chempleter-gui
    """
    favicon_path = Path(resources.files("chempleter.data").joinpath("chempleter.ico"))
    ui.run(favicon=favicon_path, reload=False, root=build_chempleter_ui)


if __name__ in {"__main__", "__mp_main__"}:
    run_chempleter_gui()
