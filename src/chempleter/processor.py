import json
import selfies as sf
import pandas as pd
from pathlib import Path


def _selfies_encoder(smiles_rep):
    """
    Encode a SMILES representation into a SELFIES representation.

    :param smiles_rep: SMILES string to encode.
    :type smiles_rep: str
    :returns: Tuple of (SELFIES string or pandas.NA, "No error" or EncoderError instance).
    :rtype: tuple
    """
    try:
        return sf.encoder(smiles_rep), "No error"
    except sf.EncoderError as e:
        return pd.NA, e


def generate_input_data(smiles_csv_path, working_dir=None):
    """
    Generate SELFIES and mapping files from a CSV file of SMILES strings.

    :param smiles_csv_path: Path to a CSV file containing a column named "smiles".
    :type smiles_csv_path: str or pathlib.Path
    :param working_dir: Directory in which output files will be written. If None, the current working directory is used.
    :type working_dir: str or pathlib.Path or None
    :raises FileNotFoundError: If the smiles CSV path or working directory do not exist.
    :raises ValueError: If the CSV file does not contain a "smiles" column.
    :returns: Tuple of file paths to cleaned selfies file, stoi file and itos file
    :rtype: tuple
    """

    smiles_path = Path(smiles_csv_path)

    if working_dir is not None:
        working_dir = Path(working_dir)
    else:
        working_dir = Path().cwd()

    if not working_dir.exists():
        raise FileNotFoundError(working_dir, "  not found.")

    if smiles_path.exists():
        df = pd.read_csv(smiles_path)
        if "smiles" in df.columns:
            df["selfies"], df["selfies_encode_error"] = zip(
                *df["smiles"].apply(_selfies_encoder)
            )
        else:
            raise ValueError("Column `smiles` not found in the CSV file.")
    else:
        raise FileNotFoundError(smiles_csv_path, "  not found.")

    df.to_csv(working_dir / "seflies_raw.csv")

    # drop all for which selfies encoding gave an error
    df_clean = df.dropna()
    df_clean.to_csv(working_dir / "selfies_clean.csv")

    selfies_list = df_clean["selfies"].to_list()
    alphabet = sf.get_alphabet_from_selfies(selfies_list)
    alphabet = ["[PAD]", "[START]", "[END]"] + list(sorted(alphabet))
    selfies_to_integer = dict(zip(alphabet, range(len(alphabet))))  # stoi file
    integer_to_selfies = list(selfies_to_integer.keys())  # itos file

    with open(working_dir / "stoi.json", "w") as f:
        json.dump(selfies_to_integer, f)
    with open(working_dir / "itos.json", "w") as f:
        json.dump(integer_to_selfies, f)

    selfies_file = working_dir / "selfies_clean.csv"
    stoi_file = working_dir / "stoi.json"
    itos_file = working_dir / "itos.json"

    return selfies_file, stoi_file, itos_file
