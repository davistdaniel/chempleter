import io
import base64
import logging
import torch
from nicegui import ui
from chempleter.inference import handle_prompt, extend, evolve, bridge, decorate
from chempleter.descriptors import calculate_descriptors
from pathlib import Path
from rdkit.Chem import Draw
from rdkit.Chem import MolFromSmiles, rdFMCS
from rdkit import Chem
from importlib import resources
from chempleter import __version__

logger = logging.getLogger(__name__)

def build_chempleter_ui():
    """
    Build Chempleter GUI using Nicegui. This fucntion also reads in the trained model, vocabulary files.
    """
    logger.info("Starting Chempleter GUI...")

    # load data
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    logger.info(f"Selected device : {device}")

    def _validate_smiles(smiles, frag1_smiles=None, frag2_smiles=None):
        try:
            prompt = handle_prompt(
                smiles,
                selfies=None,
                stoi=None,
                alter_prompt=alter_prompt_checkbox.value,
                frag1_smiles=frag1_smiles,
                frag2_smiles=frag2_smiles,
            )
            return True, prompt
        except Exception as e:
            print(e)
            return False, ""

    def _draw_if_bridge(generated_molecule):
        frag1_smiles = smiles_input.value
        frag2_smiles = smiles_input2.value

        frag1_structure = MolFromSmiles(frag1_smiles)
        frag2_structure = MolFromSmiles(frag2_smiles)

        if frag1_structure is None or frag2_structure is None:
            return Draw.MolToImage(generated_molecule, size=(300, 300))

        frag_atoms = set()

        matches1 = generated_molecule.GetSubstructMatches(frag1_structure)
        matches2 = generated_molecule.GetSubstructMatches(frag2_structure)

        for match in matches1 + matches2:
            for atom_idx in match:
                frag_atoms.add(atom_idx)

        if not frag_atoms:
            return Draw.MolToImage(generated_molecule, size=(300, 300))

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
            size=(300, 300),
            highlightAtoms=bridge_atoms,
            highlightBonds=bridge_bonds,
        )

        return img

    def _draw_if_extend(generated_molecule):
        # try highlighting input molecule
        input_molecule_structure = MolFromSmiles(smiles_input.value)
        if input_molecule_structure is None:
            return Draw.MolToImage(generated_molecule, size=(300, 300))

        match = generated_molecule.GetSubstructMatch(input_molecule_structure)
        if match is None:
            return Draw.MolToImage(generated_molecule, size=(300, 300))

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

            mcs = rdFMCS.FindMCS(
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
            subImgSize=(300, 300),
            highlightAtomLists=highlight_atoms,
            highlightBondLists=highlight_bonds,
        )

        return img

    async def show_generated_molecule():
        generated_smiles = None
        if generation_type_radio.value != "Bridge":
            if _validate_smiles(smiles=smiles_input.value)[0] is False:
                ui.notify("Error parsing input SMILES", type="negative")
                return
        else:
            if smiles_input.value == "" or smiles_input2.value == "":
                ui.notify(
                    "For bridging, two input smiles must be provided.", type="negative"
                )
                return
            if (
                _validate_smiles(smiles=smiles_input.value)[0] is False
                or _validate_smiles(smiles=smiles_input2.value)[0] is False
            ):
                ui.notify("Error parsing input SMILES", type="negative")
                return

        smiles_input.disable()
        generate_button.set_enabled(False)

        # set parameters from gui
        length = length_slider.value
        max_len = int(length * 100)
        min_len = int((length / 2) * 100)
        next_atom_criteria = (
            "greedy" if sampling_radio.value == "Most probable" else "top_k_temperature"
        )

        gen_func_dict = {"Extend": extend, "Evolve": evolve, "Bridge": bridge, "Decorate": decorate}

        gen_func = gen_func_dict[generation_type_radio.value]

        if generation_type_radio.value in ["Extend","Evolve"]:
            # generate
            generated_molecule, generated_smiles, _ = gen_func(
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

        elif generation_type_radio.value == "Decorate":
            if smiles_input.value == "":
                ui.notify("SMILES input cannot be empty for decoration",type="negative")
                smiles_input.enable()
                generate_button.set_enabled(True)
                return
            
            elif smiles_input.value!="" and atom_idx_select.value is None:
                ui.notify("Consult the figure shown and select an atom index from the dropdown, then click on GENERATE again.",type="info", multi_line=True)
                
                smiles_input.enable()
                generate_button.set_enabled(True)
                molecule_image.set_visibility(True)

                logger.info("Visualizing input molecule for decoration.")
                intial_mol = MolFromSmiles(smiles_input.value)
                for atom in intial_mol.GetAtoms():
                    atom.SetProp("atomNote", str(atom.GetIdx()))

                img = Draw.MolToImage(intial_mol, size=(400, 300))
                buffer1 = io.BytesIO()
                img.save(buffer1, format="PNG")
                img_base64 = base64.b64encode(buffer1.getvalue()).decode()
                molecule_image.style("width: 300px")
                molecule_image.set_source(f"data:image/png;base64,{img_base64}")
                logger.info("Setting atom indices")       
                atom_idx_select.set_options(list(range(intial_mol.GetNumAtoms())))
                return
            
            elif smiles_input.value != "" and atom_idx_select.value is not None:
                try:
                    generated_molecule,generated_smiles,_=decorate(smiles=smiles_input.value,atom_idx=atom_idx_select.value)
                except ValueError as e:
                    ui.notify(str(e),type="negative")
    
        elif generation_type_radio.value == "Bridge":
            generated_molecule, generated_smiles, _ = gen_func(
                frag1_smiles=smiles_input.value,
                frag2_smiles=smiles_input2.value,
                model=None,
                stoi_file=None,
                itos_file=None,
                temperature=temperature_input.value,
                next_atom_criteria=next_atom_criteria,
            )

        if generated_smiles is not None:
            # check if same
            if isinstance(generated_smiles, list):
                if len(generated_smiles) == 1:
                    ui.notify(
                        "Same molecule as input. Try increasing temperature.",
                        type="warning",
                    )
            else:
                if generated_smiles == smiles_input.value:
                    if alter_prompt_checkbox.value is False:
                        ui.notify(
                            "Same molecule as input. Try allowing prompt modification.",
                            type="warning",
                        )

            if generation_type_radio.value == "Extend":
                img = _draw_if_extend(generated_molecule)
                molecule_image.style("width: 300px")
            elif generation_type_radio.value == "Decorate":
                img = _draw_if_extend(generated_molecule)
                molecule_image.style("width: 300px")
            elif generation_type_radio.value == "Bridge":
                img = _draw_if_bridge(generated_molecule)
                molecule_image.style("width: 300px")
            else:
                img = _draw_if_evolve(generated_molecule)
                molecule_image.style(f"width: {180 * len(generated_smiles)}px")
            atom_idx_select.set_options(options=[])
            # generated image
            generated_smiles_label.set_text(
                " --> ".join(generated_smiles)
                if isinstance(generated_smiles, list)
                else generated_smiles
            )

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            molecule_image.set_source(f"data:image/png;base64,{img_base64}")
            molecule_image.set_visibility(True)
            calculated_descriptors = calculate_descriptors(
                generated_molecule[-1]
                if isinstance(generated_molecule, list)
                else generated_molecule
            )
            mw_chip.set_text(f"MW: {calculated_descriptors['MW']}")
            logp_chip.set_text(f"LogP: {calculated_descriptors['LogP']}")
            SA_score_chip.set_text(f"SA score: {calculated_descriptors['SA_Score']}")
            qed_chip.set_text(f"QED: {calculated_descriptors['QED']}")
            fsp3_chip.set_text(f"Fsp3: {calculated_descriptors['Fsp3']}")
            # tpsa_chip.set_text(f"TPSA: {calculated_descriptors["TPSA"]}")
            smiles_input.enable()
            generate_button.set_enabled(True)
        else:
            smiles_input.enable()
            generate_button.set_enabled(True)
            return

    def handle_generation_type_change():
        if generation_type_radio.value == "Bridge":
            help_expand.set_text("Bridge")
            help_label.set_content("""
                                   <b>Bridge</b>
                                   
                                   - Two molecular fragments will be bridged. Bridge will be highlighted in the output. Enter two valid SMILES to get started.
                                   
                                   """)

            smiles_input2.set_visibility(True)
            length_slider.disable()
            alter_prompt_checkbox.disable()
            temperature_input.props("max=2")
            atom_idx_select.set_visibility(False)
            atom_idx_label.set_visibility(False)
        elif generation_type_radio.value == "Extend":
            help_expand.set_text("Bridge")
            help_label.set_content("""
                                    Click on Generate to start generating molecules.<br>
                                    
                                    <b>Extend</b>

                                    - Input molecular fragment will be extended. Input molecular fragment will be highlighted in the output.
                                    """)
            length_slider.enable()
            alter_prompt_checkbox.enable()
            smiles_input2.set_visibility(False)
            temperature_input.props("max=5")
            atom_idx_select.set_visibility(False)
            atom_idx_label.set_visibility(False)
        elif generation_type_radio.value == "Evolve":
            help_expand.set_text("Evolve")
            help_label.set_content("""
                                   <b>Evolve</b>
                                   
                                   - Input molecular fragment will be evolved. New fragments added at each evolution step will be highlighted in the output.
                                   
                                   """)
            length_slider.enable()
            alter_prompt_checkbox.disable()
            smiles_input2.set_visibility(False)
            temperature_input.props("max=5")
            atom_idx_select.set_visibility(False)
            atom_idx_label.set_visibility(False)

        elif generation_type_radio.value == "Decorate":

            help_expand.set_text("Decorate")

            help_label.set_content("""
                                   <b>Decorate</b>
                                   
                                   - Input molecular fragment will be decorated at the selected atom index. Input molecular fragment will be highlighted in the output. 
                                   - Input the molecule you want to decorate as SMILES and click on GENERATE.
                                   - This will draw the input molecule with indices and populate the atom index dropdown. Select the atom index to attach the decoration to and click on GENERATE again.
                                   
                                   """)

            length_slider.disable()
            alter_prompt_checkbox.disable()
            smiles_input2.set_visibility(False)
            temperature_input.props("max=2")
            atom_idx_select.set_visibility(True)
            atom_idx_label.set_visibility(True)


    logo_path = Path(resources.files("chempleter.data").joinpath("chempleter_logo.png"))
    ui.page_title("Chempleter")
    with ui.column().classes("w-full min-h-screen items-center overflow-auto py-8"):
        with ui.row(wrap=False).classes("w-128 justify-center"):
            with ui.link(target="https://github.com/davistdaniel/chempleter"):
                ui.image(logo_path).classes("w-32")
        with ui.row().classes("w-128 justify-center"):
            with ui.expansion("Extend",icon="help").classes("w-128") as help_expand:
                help_label = ui.markdown("""
                                        Click on Generate to start generating molecules.<br>
                                        
                                        <b>Extend</b>

                                        - Input molecular fragment will be extended. Input molecular fragment will be highlighted in the output.
                                        """)
        with ui.card().tight():
            with ui.row(wrap=False).classes("w-128 justify-center"):
                smiles_input = ui.input(
                    "Enter SMILES",
                    placeholder="c1ccccc1",
                    validation=lambda value: "Invalid SMILES"
                    if _validate_smiles(smiles=value)[0] is False
                    else None,
                )
                smiles_input2 = ui.input(
                    "Enter SMILES 2",
                    placeholder="c1ccccc1",
                    validation=lambda value: "Invalid SMILES"
                    if _validate_smiles(smiles=value)[0] is False
                    else None,
                )
                smiles_input2.set_visibility(False)

            with ui.row(wrap=True).classes("w-full justify-center items-center"):
                ui.separator()
                alter_prompt_checkbox = ui.checkbox("Allow prompt modification")
                temperature_input = ui.number(
                    "Temperature", precision=1, value=0.7, step=0.1, min=0.2, max=5
                ).classes("w-32")
                atom_idx_label = ui.label("Atom index: ")
                atom_idx_label.set_visibility(False)
                atom_idx_select = ui.select(options=[])
                atom_idx_select.set_visibility(False)
                ui.separator()
            with ui.row(wrap=False):
                ui.chip("Sampling: ", color="white")
                sampling_radio = ui.radio(
                    ["Most probable", "Random"], value="Random"
                ).props("inline")
            with ui.row(wrap=False).classes("w-135 left"):
                ui.chip("Generation type: ", color="white")
                generation_type_radio = ui.radio(
                    ["Extend", "Evolve", "Bridge", "Decorate"],
                    value="Extend",
                    on_change=handle_generation_type_change,
                ).props("inline")
            with ui.row(wrap=False).classes("w-128"):
                ui.chip("Molecule size: ", color="white")
                ui.chip("Smaller")
                length_slider = ui.slider(min=0.1, max=1, step=0.05, value=0.5)
                ui.chip("Larger")

        with ui.row().classes("w-128 justify-center"):
            generate_button = ui.button("Generate", on_click=show_generated_molecule)

        with ui.row():
            mw_chip = ui.chip("MW", color="blue-3").tooltip(
                "Molecular weight including hydrogens"
            )
            logp_chip = ui.chip("LogP", color="green-3").tooltip(
                "Octanol-Water Partition Coeffecient"
            )
            SA_score_chip = ui.chip("SA score", color="orange-3").tooltip(
                "Synthetic Accessibility score ranging from 1 (easy) to 10 (difficult)"
            )
            qed_chip = ui.chip("QED", color="grey-3").tooltip(
                "Quantitative Estimate of Drug-likeness ranging from 0 to 1."
            )
            fsp3_chip = ui.chip("Fsp3", color="pink-3").tooltip(
                "Fraction of sp3 Hybridized Carbons"
            )
            # tpsa_chip = ui.chip("TPSA",color='violet-3').tooltip("Topological polar surface area")
        with ui.row().classes("w-128 justify-center"):
            generated_smiles_label = ui.label("").style(
                "font-weight: normal; color: black; font-size: 10px;"
            )   
        with ui.card(align_items="center").tight().classes("w- 256 justify-center"):
            molecule_image = ui.image()
            molecule_image.set_visibility(False)

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
