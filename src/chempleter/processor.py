import json
import selfies as sf
import pandas as pd
from pathlib import Path

def selfies_encoder(smiles_rep):
    """
    This functions encodes a SMILES representation into SELFIES representation
    
    :param smiles_rep: Description
    """
    try:
        return sf.encoder(smiles_rep), "No error"
    except sf.EncoderError as e:
        return pd.NA, e


def generate_input_data(smiles_csv_path,working_dir=None):
    """
    This function takes a single csv files containing SMILES
    
    :param smiles_csv_file: str, path to csv files containing SMILES
    """

    smiles_path = Path(smiles_csv_path)

    if working_dir is not None:
        working_dir = Path(working_dir)
    else:
        working_dir = Path().cwd()

    if not working_dir.exists():
        raise FileNotFoundError(working_dir,"  not found.")

    if smiles_path.exist():
        df = pd.read_csv(smiles_path)
        if "smiles" in df.columns():
            df["selfies"], df["selfies_encode_error"] = zip(*df["smiles"].apply(selfies_encoder))
        else:
            raise ValueError("Column `smiles` not found in the CSV file.")
    else:
        raise FileNotFoundError(smiles_csv_path,"  not found.")
    
    df.to_csv(working_dir / "seflies_raw.csv")
    
    df_clean = df.dropna()
    df_clean.to_csv(working_dir / "selfies_clean.csv")

    selfies_list = df_clean["selfies"].to_list()
    alphabet = sf.get_alphabet_from_selfies(selfies_list)
    alphabet = ["[PAD]","[START]","[END]"]+list(sorted(alphabet))
    selfies_to_integer = dict(zip(alphabet,range(len(alphabet)))) #stoi file
    integer_to_selfies = list(selfies_to_integer.keys()) #itos file

    with open(working_dir / "stoi.json","w") as f:
        json.dump(selfies_to_integer,f)
    with open("itos.json","w") as f:
        json.dump(working_dir / integer_to_selfies,f)