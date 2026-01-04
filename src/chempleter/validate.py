import torch
import logging
import json
import math
import collections
import datetime
import selfies as sf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# chempleter
from chempleter.model import ChempleterModel
from chempleter.inference import extend, bridge
from chempleter.descriptors import calculate_descriptors

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _load_checkpoint(checkpoint_path, model, device=device):
    """
    Load a model checkpoint from the specified path and restore its state.

    This function loads a PyTorch checkpoint and applies the model state dictionary
    to the provided model. It supports checkpoints saved in two formats:
    1. Direct model state dictionary (OrderedDict), applicable when checkpoint contains other state dicts
    2. Dictionary containing a 'model_state_dict' key, applicable when only model state_dict is saved

    :param checkpoint_path: Path to the checkpoint file to load.
    :type checkpoint_path: str
    :param model: The PyTorch model to load the checkpoint into.
    :type model: torch.nn.Module
    :param device: The device to load the checkpoint onto
    :type device: torch.device
    :return: The model with loaded state dictionary.
    :rtype: torch.nn.Module
    :raises FileNotFoundError: If the checkpoint file does not exist at the specified path.
    :raises ValueError: If the checkpoint format is not recognized or does not contain a valid model state dictionary.

    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path} for validation.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # check if the checpoint only has model_state_dict or alos other state_dicts
        if isinstance(checkpoint, collections.OrderedDict):
            model.load_state_dict(checkpoint)
            logging.info("Loaded model state dict from checkpoint.")
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info("Loaded model state dict from checkpoint.")
        else:
            raise ValueError(
                f"Model state_dict could not be loaded from checkpoint at path : {checkpoint_path}"
            )
    else:
        raise FileNotFoundError(
            f"checkpoint file not found at path: {str(checkpoint_path)}"
        )

    return model


def _compute_selfies_encoding_fidelity(generated_selfies, generated_smiles):
    """
    This function measures how many tokens are lost when generated SELFIES tokens are converted to SMILES.
    A higher fidelity score indicates better preservation of the original tokens.

    :param generated_selfies: List of SELFIES strings that were originally generated.
    :type generated_selfies: list of str
    :param generated_smiles: List of SMILES strings corresponding to the generated SELFIES.
    :type generated_smiles: list of str
    :return: A tuple containing:
        - fidelity_score (float): The average proportion of tokens ignored across all SELFIES strings.
          Values closer to 0 indicate better fidelity.
        - ignored_token_prop (list of float): Individual token loss proportions for each SELFIES string.
    :rtype: tuple of (float, list of float)
    """
    re_encoded_selfies_list = [sf.encoder(smiles) for smiles in generated_smiles]
    ignored_token_prop = [
        (len(i) - len(j)) / len(i)
        for i, j in zip(generated_selfies, re_encoded_selfies_list)
    ]

    return sum(ignored_token_prop) / len(ignored_token_prop), ignored_token_prop


def _compute_uniqueness(generated_smiles):
    """
    Compute the uniqueness of generated SMILES strings.

    Calculates the proportion of unique SMILES strings in the generated smiles
    by dividing the count of distinct SMILES by the total count of SMILES.

    :param generated_smiles: A collection of SMILES strings to evaluate for uniqueness.
    :type generated_smiles: list or iterable
    :return: The uniqueness ratio as a decimal between 0 and 1, where 1 represents
             100% uniqueness (all SMILES are distinct).
    :rtype: float
    """
    logging.info("Computing uniqueness of generated smiles ...")
    return len(set(generated_smiles)) / len(generated_smiles)


def _compute_novelty(generated_smiles, reference_smiles):
    """
    Compute the novelty of generated SMILES strings.

    This function calculates the proportion of unique generated SMILES strings
    that are not present in the reference (training) set. Novelty is a measure of how many
    new or previously unseen molecules were generated.

    :param generated_smiles: A list of generated SMILES.
    :type generated_smiles: list of str
    :param reference_smiles: A list of reference SMILES strings to compare against.
    :type reference_smiles: list of str
    :return: The novelty score, calculated as the ratio of novel SMILES to unique
             generated SMILES. Returns a value between 0 and 1, where 1 indicates
             all generated SMILES are novel.
    :rtype: float
    """
    logging.info("Computing novelty of generated smiles ...")
    novel_smiles = [i for i in set(generated_smiles) if i not in set(reference_smiles)]

    return len(novel_smiles) / len(set(generated_smiles))


def _compute_descriptors(generated_mols):
    """
    Compute descriptors for a a list of generated molecules and calculate aggregate statistics.

    This function calculates molecular descriptors for each molecule (rdkit.Chem.Mol) in the input list,
    then aggregates the results to compute average, minimum, and maximum values for
    each descriptor across all molecules.

    :param generated_mols: A list of rdkit.Chem.Mol objects
    :type generated_mols: list

    :return: A tuple containing four elements:
        - descriptor_metrics (dict): A dictionary mapping each descriptor name to a list of its values across all molecules.
        - avg_dict (dict): A dictionary mapping each descriptor name to its average value.
        - min_dict (dict): A dictionary mapping each descriptor name to its minimum value.
        - max_dict (dict): A dictionary mapping each descriptor name to its maximum value.
    :rtype: tuple[dict, dict, dict, dict]
    """
    descriptor_dict_list = [calculate_descriptors(m) for m in generated_mols]

    descriptor_metrics = {i: [] for i in descriptor_dict_list[0].keys()}

    for descriptor_dict in descriptor_dict_list:
        for descriptor, value in descriptor_dict.items():
            descriptor_metrics[descriptor].append(value)

    avg_dict, max_dict, min_dict = (
        {i: None for i in descriptor_dict_list[0].keys()},
        {i: None for i in descriptor_dict_list[0].keys()},
        {i: None for i in descriptor_dict_list[0].keys()},
    )

    for descriptor, value_list in descriptor_metrics.items():
        avg_dict[descriptor] = sum(value_list) / len(value_list)
        min_dict[descriptor] = min(value_list)
        max_dict[descriptor] = max(value_list)

    return descriptor_metrics, avg_dict, min_dict, max_dict


def _make_distribution_plots(descriptor_metrics, model_name):
    """
    Create and save distribution plots for descriptor metrics.

    :param descriptor_metrics: descriptor metrics dict with metric value_list pairs
    :type descriptor_metrics: dict
    :param model_name: The name of the model, used as the base filename for
                       saving the figure (without extension).
    :type model_name: str
    :return: None
    :rtype: None
    """
    n_subplots = len(descriptor_metrics)

    n_cols = math.ceil(math.sqrt(n_subplots))
    n_rows = math.ceil(n_subplots / n_cols)

    fig, axes = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(4 * n_cols, 3 * n_rows)
    )

    axes = axes.flatten()

    for ax, (metric, value) in zip(axes, descriptor_metrics.items()):
        ax.set_title(metric + " distribution")
        ax.set_xlabel(metric)
        ax.hist(value, bins=15)

    for ax in axes[len(descriptor_metrics) :]:
        ax.remove()

    fig.tight_layout()
    fig.savefig(f"{model_name}.png", dpi=300)


def validate_checkpoint(
    checkpoint_path,
    stoi_file,
    itos_file,
    model_type,
    model_name,
    reference_smiles_path=None,
    model=None,
    n_samples=1000,
    report_format="rst",
):
    """
    Validate a trained checkpoint by generating samples and computing metrics.

    This function loads a checkpoint, generates molecules using either the 'extend' or 'bridge'
    model type, and computes various validation metrics including uniqueness, novelty, and
    molecular descriptors. It produces a formatted report and descriptor distribution plots.

    :param checkpoint_path: Path to the model checkpoint file.
    :type checkpoint_path: str
    :param stoi_file: Path to the string-to-integer (stoi) vocabulary mapping file.
    :type stoi_file: str
    :param itos_file: Path to the integer-to-string (itos) vocabulary mapping file.
    :type itos_file: str
    :param model_type: Type of generation model to use. Either 'extend' or 'bridge'. Also provide corresponding stoi, itos files.
    :type model_type: str
    :param model_name: Name of the model, used for output file naming.
    :type model_name: str
    :param reference_smiles_path: Optional path to a CSV file containing reference (training) SMILES strings
        for novelty computation. If None, novelty will not be computed. Defaults to None.
    :type reference_smiles_path: str, optional
    :param model: Optional pre-initialized ChempleterModel instance. If None, a new model
        will be created with vocabulary size determined from stoi_file. Defaults to None.
    :type model: ChempleterModel, optional
    :param n_samples: Number of molecular samples to generate for validation. Defaults to 1000.
    :type n_samples: int
    :param report_format: Format for the validation report output. Either 'rst' or 'md'.
        Defaults to 'rst'.
    :type report_format: str

    :returns: A tuple containing:
        - generated_mol_list: List of generated RDKit molecule objects.
        - generated_smiles_list: List of generated SMILES strings.
        - generated_selfies_list: List of generated SELFIES strings.
        - descriptor_metrics: Dictionary containing computed molecular descriptor metrics.
    :rtype: tuple(list, list, list, dict)

    :raises ValueError: If model_type is not 'extend' or 'bridge', or if report_format
        is not 'rst' or 'md'.
    """

    with open(stoi_file) as f:
        stoi = json.load(f)

    if model is None:
        model = ChempleterModel(vocab_size=len(stoi))

    model = _load_checkpoint(checkpoint_path=checkpoint_path, model=model)

    logging.info(f"Generating {n_samples} samples for validation...")
    logger = logging.getLogger()
    logger.setLevel("ERROR")

    generated_mol_list, generated_smiles_list, generated_selfies_list = [], [], []

    for idx in tqdm(range(n_samples)):
        if model_type == "extend":
            m, smiles, selfies = extend(
                smiles="",
                model=model,
                stoi_file=stoi_file,
                itos_file=itos_file,
                next_atom_criteria="temperature",
                alter_prompt=False,
                randomise_prompt=False,
                temperature=0.7,
                k=10,
                min_len=None,
                max_len=50,
            )
        elif model_type == "bridge":
            m, smiles, selfies = bridge(
                frag1_smiles="c1ccccc1",
                frag2_smiles="c1ccccc1",
                model=model,
                stoi_file=stoi_file,
                itos_file=itos_file,
                temperature=1,
                next_atom_criteria="temperature",
                k=10,
            )
        else:
            raise ValueError("Invalid model type")

        generated_mol_list.append(m)
        generated_smiles_list.append(smiles)
        generated_selfies_list.append(selfies)

    logger = logging.getLogger()
    logger.setLevel("INFO")

    logging.info(f"Generated {n_samples} samples for validation.")

    if reference_smiles_path is not None:
        # make a list of reference smiles
        try:
            reference_smiles = pd.read_csv(Path(reference_smiles_path))[
                "smiles"
            ].to_list()
        except Exception as e:
            reference_smiles = None
            logging.error(
                "Unabe to load reference smiles, novelty won't be computed: ", e
            )
    else:
        reference_smiles = None
        logging.info(
            "novelty won't be computed, since no reference smiles path was given."
        )

    avg_ignored_token_fraction, ignored_token_prop = _compute_selfies_encoding_fidelity(
        generated_selfies=generated_selfies_list, generated_smiles=generated_smiles_list
    )
    uniqueness = _compute_uniqueness(generated_smiles_list)
    novelty = (
        _compute_novelty(
            generated_smiles=generated_smiles_list, reference_smiles=reference_smiles
        )
        if reference_smiles is not None
        else None
    )

    print(
        f"Average fraction of tokens ignored by SELFIES decoding : {avg_ignored_token_fraction:.2f}"
    )
    print(f"Fraction of unique generated smiles : {uniqueness:.2f}")
    if novelty:
        print(
            f"Fraction of generated smiles which are different from the training data set: {novelty:.2f}"
        )

    descriptor_metrics, avg_dict, min_dict, max_dict = _compute_descriptors(
        generated_mols=generated_mol_list
    )

    descriptor_metrics["SELFIES decoding fidelity"] = ignored_token_prop

    logging.info("Generating distribution plots...")
    _make_distribution_plots(descriptor_metrics, model_name)

    if report_format == "rst":
        report_func = _generate_validation_report_rst
        output_path = Path.cwd() / f"{model_name}.rst"
    elif report_format == "md":
        report_func = _generate_validation_report
        output_path = Path.cwd() / f"{model_name}.md"
    else:
        raise ValueError(f"Invalid report format type: {report_format}")

    report_func(
        output_path=output_path,
        model_name=model_name,
        n_samples=n_samples,
        uniqueness=uniqueness,
        novelty=novelty,
        selfies_fidelity_avg=avg_ignored_token_fraction,
        descriptor_avg=avg_dict,
        descriptor_max=max_dict,
        descriptor_min=min_dict,
        distribution_figure_path=f"{model_name}.png",
    )

    return (
        generated_mol_list,
        generated_smiles_list,
        generated_selfies_list,
        descriptor_metrics,
    )


def _generate_validation_report(
    output_path,
    model_name,
    n_samples,
    uniqueness,
    novelty,
    selfies_fidelity_avg,
    descriptor_avg,
    descriptor_min,
    descriptor_max,
    distribution_figure_path,
):
    """
    Generate a validation report in markdown format.

    This function creates a comprehensive validation report for a generative model,
    including generation quality metrics, descriptor statistics, and distribution
    visualizations. The report is written to a file in RST format.

    :param output_path: Path where the md (markdown) validation report will be saved.
    :type output_path: str
    :param model_name: Name of the model being validated.
    :type model_name: str
    :param n_samples: Number of samples to be generated by the model for validation.
    :type n_samples: int
    :param uniqueness: Uniqueness metric of generated samples (0-1 scale).
    :type uniqueness: float
    :param novelty: Novelty metric of generated samples (0-1 scale) or None.
    :type novelty: float or None
    :param selfies_fidelity_avg: Average SELFIES fidelity score of generated samples.
    :type selfies_fidelity_avg: float
    :param descriptor_avg: Dictionary mapping descriptor names to their average values.
    :type descriptor_avg: dict
    :param descriptor_min: Dictionary mapping descriptor names to their minimum values.
    :type descriptor_min: dict
    :param descriptor_max: Dictionary mapping descriptor names to their maximum values.
    :type descriptor_max: dict
    :param distribution_figure_path: Path to the descriptor distribution visualization image.
    :type distribution_figure_path: str
    :return: Path to the generated validation report file.
    :rtype: Path
    """
    output_path = Path(output_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    metrics = {
        "Uniqueness": uniqueness,
        "Novelty": novelty if novelty is not None else float("nan"),
        "SELFIES fidelity": selfies_fidelity_avg,
    }
    metrics_header = "| " + " | ".join(metrics.keys()) + " |\n"
    metrics_header += "| " + " | ".join(["---"] * len(metrics)) + " |\n"
    metrics_row = (
        "| "
        + " | ".join(
            f"{v:.4f}" if not isinstance(v, float) or not v != v else "N/A"
            for v in metrics.values()
        )
        + " |\n"
    )  # nan -> N/A

    descriptors = descriptor_avg.keys()
    descriptor_header = (
        "| Descriptor | Average | Minimum | Maximum |\n|---|---|---|---|\n"
    )
    descriptor_rows = ""
    for desc in descriptors:
        descriptor_rows += f"| {desc} | {descriptor_avg[desc]:.4f} | {descriptor_min[desc]:.4f} | {descriptor_max[desc]:.4f} |\n"

    md = f"""
# Model Validation Report

**Model:** {model_name}  
**Samples Generated:** {n_samples}  
**Generated On:** {timestamp}

---

## Generation Quality Metrics

{metrics_header}{metrics_row}

---

## Descriptor Statistics

{descriptor_header}{descriptor_rows}

---

## Descriptor Distributions

![Descriptor Distributions]({distribution_figure_path})

---

## Notes

---
"""

    output_path.write_text(md.strip())
    return output_path


def _generate_validation_report_rst(
    output_path,
    model_name,
    n_samples,
    uniqueness,
    novelty,
    selfies_fidelity_avg,
    descriptor_avg,
    descriptor_min,
    descriptor_max,
    distribution_figure_path,
):
    """
    Generate a validation report in reStructuredText format.

    This function creates a comprehensive validation report for a generative model,
    including generation quality metrics, descriptor statistics, and distribution
    visualizations. The report is written to a file in RST format.

    :param output_path: Path where the RST validation report will be saved.
    :type output_path: str
    :param model_name: Name of the model being validated.
    :type model_name: str
    :param n_samples: Number of samples to be generated by the model for validation.
    :type n_samples: int
    :param uniqueness: Uniqueness metric of generated samples (0-1 scale).
    :type uniqueness: float
    :param novelty: Novelty metric of generated samples (0-1 scale) or None.
    :type novelty: float or None
    :param selfies_fidelity_avg: Average SELFIES fidelity score of generated samples.
    :type selfies_fidelity_avg: float
    :param descriptor_avg: Dictionary mapping descriptor names to their average values.
    :type descriptor_avg: dict
    :param descriptor_min: Dictionary mapping descriptor names to their minimum values.
    :type descriptor_min: dict
    :param descriptor_max: Dictionary mapping descriptor names to their maximum values.
    :type descriptor_max: dict
    :param distribution_figure_path: Path to the descriptor distribution visualization image.
    :type distribution_figure_path: str
    :return: Path to the generated validation report file.
    :rtype: Path
    """

    output_path = Path(output_path)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    metrics = {
        "Uniqueness": uniqueness,
        "Novelty": novelty if novelty is not None else float("nan"),
        "SELFIES fidelity": selfies_fidelity_avg,
    }

    col_widths = [max(len(k), 10) for k in metrics.keys()]
    metrics_values = []
    for v in metrics.values():
        metrics_values.append(
            f"{v:.4f}" if not isinstance(v, float) or not v != v else "N/A"
        )
        col_widths[metrics_values.index(metrics_values[-1])] = max(
            col_widths[metrics_values.index(metrics_values[-1])],
            len(metrics_values[-1]),
        )

    # generation quality metrics
    def rst_table(headers, rows):
        # column widths have to be calcualted
        widths = [
            max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]
        sep = " ".join("=" * w for w in widths)
        header_row = " ".join(h.ljust(w) for h, w in zip(headers, widths))
        data_row = " ".join(r.ljust(w) for r, w in zip(rows, widths))
        table = f"{sep}\n{header_row}\n{sep}\n{data_row}\n{sep}\n"
        return table

    generation_table_rst = rst_table(list(metrics.keys()), metrics_values)

    # descriptor statistics
    descriptor_headers = ["Descriptor", "Average", "Minimum", "Maximum"]
    descriptor_rows = []
    for desc in descriptor_avg.keys():
        row = [
            desc,
            f"{descriptor_avg[desc]:.4f}",
            f"{descriptor_min[desc]:.4f}",
            f"{descriptor_max[desc]:.4f}",
        ]
        descriptor_rows.append(row)

    # column widths for descriptor table
    col_widths_desc = [
        max(len(h), max(len(r[i]) for r in descriptor_rows))
        for i, h in enumerate(descriptor_headers)
    ]
    sep_desc = " ".join("=" * w for w in col_widths_desc)
    header_desc = " ".join(
        h.ljust(w) for h, w in zip(descriptor_headers, col_widths_desc)
    )
    descriptor_table_rst = sep_desc + "\n" + header_desc + "\n" + sep_desc + "\n"
    for row in descriptor_rows:
        descriptor_table_rst += (
            " ".join(r.ljust(w) for r, w in zip(row, col_widths_desc)) + "\n"
        )
    descriptor_table_rst += sep_desc + "\n"

    # add genrated image
    image_rst = f".. image:: {distribution_figure_path}\n   :alt: Descriptor Distributions\n   :align: center\n"

    title = f"Validation report: {model_name}"
    underline = "=" * len(title)

    # final rst
    rst_content = f"""
{title}
{underline}

Model: {model_name}

Samples Generated: {n_samples}  

Generated On: {timestamp}

Generation Quality Metrics
--------------------------

{generation_table_rst}

Descriptor Statistics
---------------------

{descriptor_table_rst}

Descriptor Distributions
------------------------

{image_rst}
"""

    output_path.write_text(rst_content.strip())
    return output_path
