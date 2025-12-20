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


def generate_input_data(smiles_csv_path):
    """
    This function takes a single csv files containing SMILES
    
    :param smiles_csv_file: str, path to csv files containing SMILES
    """

    smiles_path = Path(smiles_csv_path)

    if smiles_path.exist():
        df = pd.read_csv(smiles_path)
        if "smiles" in df.columns():
            df["selfies"], df["selfies_encode_error"] = zip(*df["smiles"].apply(selfies_encoder))
        else:
            raise ValueError("Column `smiles` not found in the CSV file.")
    else:
        raise FileNotFoundError(smiles_csv_path,"  not found.")
    
    df.to_csv("seflies_data_raw.csv")
    
    df_clean = df.dropna()
    df_clean.to_csv("selfies_data_clean.csv")

    selfies_list = df_clean["selfies"].to_list()
    alphabet = sf.get_alphabet_from_selfies(selfies_list)
    alphabet = ["[PAD]","[START]","[END]"]+list(sorted(alphabet))
    selfies_to_integer = dict(zip(alphabet,range(len(alphabet)))) #stoic file
    integer_to_selfies = list(selfies_to_integer.keys()) #itos file

    with open("selfies_to_integer.json","w") as f:
        json.dump(selfies_to_integer,f)
    with open("integer_to_selfies.json","w") as f:
        json.dump(integer_to_selfies,f)