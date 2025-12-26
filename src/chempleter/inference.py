import json
import torch
import selfies as sf
from rdkit import Chem
import random
import logging
from pathlib import Path
from importlib import resources
from chempleter.model import ChempleterModel


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def handle_prompt(smiles, selfies, stoi=None, alter_prompt=False):
    """
    This function handles the smiles input by the user.
    Note that the selfies can be directly given, in that case, it will just add the "[START]" token.
    If neither selfies nor smiles is given, the "[START]" token is used as prompt.

    :param smiles: input SMILES
    :type smiles: str
    :param selfies: A list of SELFIES tokens
    :type selfies: list
    :param stoi: strings to integer mapping
    :type stoi: dict
    :param alter_prompt: Flag for prompt modification during input and generation
    :type alter_prompt: bool
    """
    if stoi is None:
        stoi_file = Path(resources.files("chempleter.data").joinpath("stoi.json"))
        with open(stoi_file) as f:
            stoi = json.load(f)

    if selfies is not None:
        logging.info(f"Input SELFIES: {smiles}")
        for token in selfies:
            if token not in stoi.keys():
                raise ValueError("Invalid SELFIES Token: ", token)
        # manual tokens, only start token is added in this case.
        prompt = ["[START]"] + selfies
    else:
        if smiles.strip().replace(" ", "") != "":
            logging.info(f"Input SMILES: {smiles}")
            try:
                _ = sf.encoder(smiles)
                test_smiles = smiles
            except sf.EncoderError as e:
                logging.error("SMILES encode error.")
                if alter_prompt:
                    logging.debug("alter_prompt is True, altering prompt.")
                    # start removing characters from end.
                    for i in range(len(smiles), 0, -1):
                        try:
                            test_smiles = smiles[:i]
                            logging.info(f"Altered SMILES: {test_smiles}")
                            _ = sf.encoder(test_smiles)
                            tail = smiles[i:]
                            if len(tail) > 0:
                                logging.info(f"Ingored string: {tail}")
                            break
                        except sf.EncoderError:
                            logging.error("SMILES encode error.")
                            continue
                    raise sf.EncoderError(e)
                else:
                    raise sf.EncoderError(e)

            prompt = ["[START]"] + list(
                sf.split_selfies(sf.encoder(test_smiles, strict=False))
            )
        else:
            logging.debug(
                "No input for smiles or selfies, using default prompt : [START]"
            )
            prompt = ["[START]"]

    return prompt


def handle_len(prompt, min_len, max_len):
    """
    Adjust minimum and maximum generated-token lengths relative to the prompt length.

    :param prompt: Input prompt used to compute its length.
    :type prompt: str
    :param min_len: Desired minimum generated tokens; if None or less than prompt length it is set to prompt_len + 2.
    :type min_len: int or None
    :param max_len: Desired maximum generated tokens (treated as additional tokens); prompt length is added to produce an absolute max.
    :type max_len: int
    :returns: Tuple of adjusted (min_len, max_len) as absolute token counts.
    :rtype: tuple[int, int]
    """
    prompt_len = len(prompt)

    if min_len is None:
        min_len = prompt_len + 2
        logging.debug(f"min_len is None. Setting min_len to {min_len}")

    if min_len < prompt_len:
        logging.warning(
            f"min_len ({min_len}) < prompt length ({prompt_len}). "
            f"Setting min_len to {prompt_len + 2}"
        )
        min_len = prompt_len + 2

    max_len += prompt_len

    if max_len < min_len:
        max_len + 5
        logging.debug(f"max_len < min_len; Setting max_len to {max_len}")

    logging.info(
        f"min_len = {min_len}, max_len = {max_len}, prompt_len = {prompt_len}."
    )
    return min_len, max_len


def handle_sampling(last_atom_logits, next_atom_criteria, temperature, k):
    """
    Decide how the next token is sampled.

    :param last_atom_logits: Logits for the last predicted atom.
    :type last_atom_logits: torch.Tensor
    :param next_atom_criteria: Sampling strategy; one of "greedy", "temperature", "top_k_temperature", "random".
    :type next_atom_criteria: str
    :param temperature: Temperature for softmax sampling.
    :type temperature: float
    :param k: Number of top tokens to consider for top-k sampling.
    :type k: int
    :returns: Selected next token id.
    :rtype: int
    """

    if next_atom_criteria == "random":
        next_atom_criteria = random.choice(
            ["greedy", "temperature", "top_k_temperature"]
        )
    if next_atom_criteria == "greedy":
        next_atom_id = torch.argmax(last_atom_logits).item()
    elif next_atom_criteria == "temperature":
        probs = torch.softmax(last_atom_logits / temperature, dim=-1)
        next_atom_id = torch.multinomial(probs, 1).item()
    elif next_atom_criteria == "top_k_temperature":
        topk_probs, topk_indices = torch.topk(
            torch.nn.functional.softmax(last_atom_logits / temperature, dim=-1), k
        )
        topk_probs /= topk_probs.sum()
        next_atom_id = topk_indices[torch.multinomial(topk_probs, 1)].item()
    else:
        print("Using default sampling.")
        next_atom_id = torch.argmax(last_atom_logits).item()

    return next_atom_id


def output_molecule(generated_ids, itos):
    """
    Convert generated token IDs to SELFIES string and decode to SMILES.

    :param generated_ids: Generated token IDs
    :type generated_ids: list
    :param itos: Integer to string token mapping
    :type itos: dict
    :returns: Tuple of (SMILES string, SELFIES string)
    :rtype: tuple[str, str]
    """

    generated_selfies = "".join(
        [
            itos[idx]
            for idx in generated_ids
            if itos[idx] not in ["[END]", "[START]", "[PAD]"]
        ]
    )
    logging.info(f"Generated SELFIE: {generated_selfies}")
    generated_smiles = sf.decoder(generated_selfies)
    logging.info(f"Generated SMILES from decoding: {generated_smiles}")
    ignored_token_proportion = (len(generated_selfies)-len(sf.encoder(generated_smiles)))/len(generated_selfies)
    logging.info(f"Proportion of generated tokens ignored by SELFIES decoding : {ignored_token_proportion}")


    return generated_smiles, generated_selfies


def generation_loop(
    model, prompt, stoi, min_len, max_len, next_atom_criteria, temperature, k
):
    """
    This is the main generation loop, which uses the model to produce tokens.

    :param model: Trained pytorch model
    :type model: chempleter.model.ChempleterModel
    :param prompt: Input prompt
    :type prompt: list
    :param stoi: Vocabulary or string to integer dictionary
    :type stoi: dict
    :param min_len: Minimum length of generated tokens
    :type min_len: int
    :param max_len: Maximum length of generated tokens
    :type max_len: int
    :param next_atom_criteria: Sampling strategy, one of "greedy", "temperature", "top_k_temperature", or "random"
    :type next_atom_criteria: str
    :param temperature: Temperature for softmax sampling
    :type temperature: float
    :param k: Number of top tokens to consider for top-k sampling
    :type k: int
    :returns: Generated token IDs
    :rtype: list
    """

    with torch.no_grad():
        seed_ids = [stoi[symbol] for symbol in prompt]
        generated_ids = seed_ids[:]
        current_input = torch.tensor([seed_ids]).to(device)
        
        hidden = None

        for i in range(max_len):

            current_lengths = torch.tensor([current_input.size(1)])


            logits, hidden = model(current_input, current_lengths, hidden)
            last_atom_logits = logits[0, -1, :]


            next_atom_id = handle_sampling(
                last_atom_logits, next_atom_criteria, temperature, k
            )

            if next_atom_id == stoi["[END]"]:
                if len(generated_ids) < min_len:
                    last_atom_logits[stoi["[END]"]] = -float("inf")
                    next_atom_id = handle_sampling(
                        last_atom_logits, next_atom_criteria, temperature, k
                    )
                else:
                    break

            generated_ids.append(next_atom_id)

            current_input = torch.tensor([[next_atom_id]]).to(device)

    return generated_ids

def _get_default_data(model,stoi_file,itos_file):
    default_stoi_file = Path(resources.files("chempleter.data").joinpath("stoi.json"))
    default_itos_file = Path(resources.files("chempleter.data").joinpath("itos.json"))
    default_checkpoint_file = Path(
        resources.files("chempleter.data").joinpath("model.pt")
    )

    if stoi_file is None:
        stoi_file = default_stoi_file
    if itos_file is None:
        itos_file = default_itos_file

    with open(stoi_file) as f:
        stoi = json.load(f)
    with open(itos_file) as f:
        itos = json.load(f)

    if model is None:
        model = ChempleterModel(vocab_size=len(stoi))
        checkpoint = torch.load(default_checkpoint_file, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    return stoi, itos, model

def extend(
    model=None,
    stoi_file=None,
    itos_file=None,
    selfies=None,
    smiles="",
    min_len=None,
    max_len=50,
    temperature=0.7,
    k=10,
    next_atom_criteria="top_k_temperature",
    device=device,
    alter_prompt=False,
):
    """
    Extend a molecule given a substructure.

    :param model: Trained ChempleterModel. If None, a default trained model is loaded.
    :type model: chempleter.model.ChempleterModel or None
    :param stoi_file: Path to JSON file mapping strings to integers.
    :type stoi_file: pathlib.Path or None
    :param itos_file: Path to JSON file mapping integers to strings.
    :type itos_file: pathlib.Path or None
    :param selfies: Input SELFIES tokens list (if provided, smiles is ignored).
    :type selfies: list[str] or None
    :param smiles: Input SMILES string (used if selfies is None).
    :type smiles: str
    :param min_len: Minimum number of generated tokens (absolute final length).
    :type min_len: int or None
    :param max_len: Maximum number of generated tokens (treated as additional tokens).
    :type max_len: int
    :param temperature: Sampling temperature for softmax sampling.
    :type temperature: float
    :param k: Number of top tokens to consider for top-k sampling.
    :type k: int
    :param next_atom_criteria: Sampling strategy; one of "greedy", "temperature", "top_k_temperature", "random".
    :type next_atom_criteria: str
    :param device: Device identifier to run the model on (e.g. "cpu" or accelerator type).
    :type device: str
    :param alter_prompt: Whether to allow prompt alteration if generation fails or input encoding errors occur.
    :type alter_prompt: bool

    :returns: Tuple with an RDKit molecule, generated SMILES string, and generated SELFIES string.
    :rtype: tuple[rdkit.Chem.Mol, str, str]

    :raises ValueError: If the generated molecule is invalid.
    """

    # get default data if model, stoi or itos is not given
    stoi, itos, model = _get_default_data(model,stoi_file,itos_file)

    # put model in evaluation mode
    model.to(device)
    model.eval()  

    # check prompt
    prompt = handle_prompt(smiles, selfies, stoi, alter_prompt)

    # check len
    min_len, max_len = handle_len(prompt, min_len, max_len)

    # generate
    generated_smiles = prompt
    generated_ids = generation_loop(
        model, prompt, stoi, min_len, max_len, next_atom_criteria, temperature, k
    )
    generated_smiles, generated_selfies = output_molecule(generated_ids, itos)

    retry_n = 0
    while generated_smiles == smiles and len(prompt) > 0:
        if alter_prompt:
            prompt = prompt[:-1]
            retry_n += 1
            logging.info(f"Retry {retry_n} with altered prompt : {prompt}")
            generated_ids = generation_loop(
                model,
                prompt,
                stoi,
                min_len,
                max_len,
                next_atom_criteria,
                temperature,
                k,
            )
            generated_smiles, generated_selfies = output_molecule(generated_ids, itos)
        else:
            logging.warning(
                "Same molecule as prompt. This molecule cannot be extended. Try again with a different prompt."
            )
            break

    m = Chem.MolFromSmiles(generated_smiles)
    if m is None:
        raise ValueError("Invalid molecule")


    return m, generated_smiles, generated_selfies
