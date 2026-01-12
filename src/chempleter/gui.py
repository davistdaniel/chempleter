import io
import base64
import logging
import torch
from nicegui import ui
from pathlib import Path
from importlib import resources

from chempleter import __version__
from chempleter.inference import extend, evolve, bridge, decorate
from chempleter.utils import _validate_smiles, _draw_if_bridge, _draw_if_evolve, _draw_if_extend, _draw_input_if_decorate
from chempleter.descriptors import calculate_descriptors

logger = logging.getLogger(__name__)


def build_chempleter_ui():
    """
    Build Chempleter GUI using Nicegui. This fucntion also reads in the trained model, vocabulary files.
    """
    logger.info("Starting Chempleter GUI...")

    # set device for inference, this is also set internally in each module, so might be redundant.
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    logger.info(f"Selected device : {device}")

    # handle change when generation type toggle is changed
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

            length_slider.enable()
            alter_prompt_checkbox.enable()
            smiles_input2.set_visibility(False)
            temperature_input.props("max=2")
            atom_idx_select.set_visibility(True)
            atom_idx_label.set_visibility(True)

    def show_generated_molecule():
        """
        Visualizes a generated molecule
        """
        
        generated_smiles = None

        if generation_type_radio.value != "Bridge":
            if _validate_smiles(smiles=smiles_input.value,alter_prompt_checkbox_value=alter_prompt_checkbox.value)[0] is False:
                ui.notify("Error parsing input SMILES", type="negative")
                return
        else:
            if smiles_input.value == "" or smiles_input2.value == "":
                ui.notify(
                    "For bridging, two input smiles must be provided.", type="negative"
                )
                return
            if (
                _validate_smiles(smiles=smiles_input.value,alter_prompt_checkbox_value=alter_prompt_checkbox.value)[0] is False
                or _validate_smiles(smiles=smiles_input2.value,alter_prompt_checkbox_value=alter_prompt_checkbox.value)[0] is False
            ):
                ui.notify("Error parsing input SMILES", type="negative")
                return
            
        # input smiles checks passed, disable controls for now.
        smiles_input.disable()
        generate_button.set_enabled(False)

        # set parameters from gui
        length = length_slider.value
        max_len = int(length * 100)
        min_len = int((length / 2) * 100)
        next_atom_criteria = (
            "greedy" if sampling_radio.value == "Most probable" else "top_k_temperature"
        )

        gen_func_dict = {
            "Extend": extend,
            "Evolve": evolve,
            "Bridge": bridge,
            "Decorate": decorate,
        }

        gen_func = gen_func_dict[generation_type_radio.value]

        if generation_type_radio.value in ["Extend", "Evolve"]:
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
                ui.notify(
                    "SMILES input cannot be empty for decoration", type="negative"
                )
                smiles_input.enable()
                generate_button.set_enabled(True)
                return

            elif smiles_input.value != "" and atom_idx_select.value is None: # an input smiles is given, populate atom index
                ui.notify(
                    "Consult the figure shown and select an atom index from the dropdown, then click on GENERATE again.",
                    type="info",
                    multi_line=True,
                )

                logger.info("Visualizing input molecule for decoration.")
                molecule_image.set_visibility(True)
                buffer1 = io.BytesIO()
                img, initial_mol = _draw_input_if_decorate(smiles_input.value)
                img.save(buffer1, format="PNG")
                img_base64 = base64.b64encode(buffer1.getvalue()).decode()
                molecule_image.style(
                    f"width: {300 if len(smiles_input.value) < 50 else 600}px"
                )
                molecule_image.set_source(f"data:image/png;base64,{img_base64}") #  shows the image in same space as generated molecule

                logger.info("Setting atom indices and populating atom index dropdown")
                atom_idx_select.set_options(list(range(initial_mol.GetNumAtoms())))
                atom_idx_label.style("color: green;")
                
                smiles_input.enable()
                generate_button.set_enabled(True)
                return

            elif smiles_input.value != "" and atom_idx_select.value is not None: # both input smiles given and atom index selected
                try:
                    generated_molecule, generated_smiles, _ = gen_func(
                        smiles=smiles_input.value,
                        atom_idx=atom_idx_select.value,
                        alter_prompt=alter_prompt_checkbox.value,
                        min_len=min_len,
                        max_len=max_len,
                    )
                except ValueError as e:
                    ui.notify(str(e), type="negative")

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

        # check the generated smiles 
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
            
            # visualize generated molecule, sizes need to be implemented better
            if generation_type_radio.value == "Extend":
                img = _draw_if_extend(smiles_input.value, generated_molecule)
                molecule_image.style("width: 300px")
            elif generation_type_radio.value == "Decorate":
                img = _draw_if_extend(smiles_input.value, generated_molecule)
                molecule_image.style(
                    f"width: {300 if len(smiles_input.value) < 50 else 600}px"
                )
            elif generation_type_radio.value == "Bridge":
                img = _draw_if_bridge(smiles_input.value, smiles_input2.value, generated_molecule)
                molecule_image.style("width: 300px")
            elif generation_type_radio.value == "Evolve":
                img = _draw_if_evolve(generated_molecule)
                molecule_image.style(f"width: {180 * len(generated_smiles)}px")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            molecule_image.set_source(f"data:image/png;base64,{img_base64}")
            molecule_image.set_visibility(True)

            # rest atom_idx in case decorate was used.
            atom_idx_select.set_options(options=[])
            atom_idx_label.style("color: red;")

            # display generated smiles
            generated_smiles_label.set_text(
                " --> ".join(generated_smiles)
                if isinstance(generated_smiles, list)
                else generated_smiles
            )

            # calculate and display descriptors, in case of evolvem it is the last molecule.
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
            tpsa_chip.set_text(f"TPSA: {calculated_descriptors["TPSA"]}")
            smiles_input.enable()
            generate_button.set_enabled(True)
        else:
            smiles_input.enable()
            generate_button.set_enabled(True)
            return
        

    #### Chempleter GUI ####    

    logo_path = Path(resources.files("chempleter.data").joinpath("chempleter_logo.png"))
    ui.page_title("Chempleter")

    with ui.column().classes("w-full min-h-screen items-center overflow-auto py-8"):

        #logo
        with ui.row(wrap=False).classes("w-128 justify-center"):
            with ui.link(target="https://github.com/davistdaniel/chempleter"):
                ui.image(logo_path).classes("w-32")
        
        # help label
        with ui.row().classes("w-128 justify-center"):
            with ui.expansion("Extend", icon="help").classes("w-128") as help_expand:
                help_label = ui.markdown("""
                                        Click on Generate to start generating molecules.<br>
                                        
                                        <b>Extend</b>

                                        - Input molecular fragment will be extended. Input molecular fragment will be highlighted in the output.
                  
                                        """)
        # generation input and options
        with ui.card().tight():
            with ui.row(wrap=False).classes("w-128 justify-center"):
                smiles_input = ui.input(
                    "Enter SMILES",
                    placeholder="c1ccccc1",
                    validation=lambda value: "Invalid SMILES"
                    if _validate_smiles(smiles=value,alter_prompt_checkbox_value=alter_prompt_checkbox.value)[0] is False
                    else None,
                )
                smiles_input2 = ui.input(
                    "Enter SMILES 2",
                    placeholder="c1ccccc1",
                    validation=lambda value: "Invalid SMILES"
                    if _validate_smiles(smiles=value,alter_prompt_checkbox_value=alter_prompt_checkbox.value)[0] is False
                    else None,
                )
                smiles_input2.set_visibility(False)

            with ui.row(wrap=True).classes("w-full justify-center items-center"):
                ui.separator()
                alter_prompt_checkbox = ui.checkbox("Allow prompt modification")
                temperature_input = ui.number(
                    "Temperature", precision=1, value=0.7, step=0.1, min=0.2, max=5
                ).classes("w-32")
                atom_idx_label = ui.label("Atom index: ").style("color: red;")
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

        # descriptors
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
            tpsa_chip = ui.chip("TPSA",color='violet-3').tooltip("Topological polar surface area. Values above 140 Å^2 indicate poor membrane permeability.")
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
