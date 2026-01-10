import json
import torch
import selfies as sf
from rdkit import Chem
import random
import logging
from pathlib import Path
from importlib import resources
from chempleter.model import ChempleterModel
from rdkit.Chem import RWMol, CombineMols

# logging setup
logger = logging.getLogger(__name__)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _get_default_data(generation_type, model, stoi_file, itos_file):
    if generation_type in ["extend", "evolve"]:
        default_stoi_file = Path(
            resources.files("chempleter.data").joinpath("stoi.json")
        )
        default_itos_file = Path(
            resources.files("chempleter.data").joinpath("itos.json")
        )
        default_checkpoint_file = Path(
            resources.files("chempleter.data").joinpath("model.pt")
        )

    elif generation_type == "bridge":
        default_stoi_file = Path(
            resources.files("chempleter.data").joinpath("bridge_stoi.json")
        )
        default_itos_file = Path(
            resources.files("chempleter.data").joinpath("bridge_itos.json")
        )
        default_checkpoint_file = Path(
            resources.files("chempleter.data").joinpath("bridge_model.pt")
        )

    else:
        raise ValueError("Invalid generation type")

    if stoi_file is None:
        logger.info("Using default stoi file")
        stoi_file = default_stoi_file
    if itos_file is None:
        logger.info("Using default itos file")
        itos_file = default_itos_file

    with open(stoi_file) as f:
        stoi = json.load(f)
    with open(itos_file) as f:
        itos = json.load(f)

    if model is None:
        logger.info("Using default model checkpoint")
        model = ChempleterModel(vocab_size=len(stoi))
        checkpoint = torch.load(
            default_checkpoint_file, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint)

    return stoi, itos, model


def handle_prompt(
    smiles="",
    selfies=None,
    stoi=None,
    alter_prompt=False,
    frag1_smiles=None,
    frag2_smiles=None,
):
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
        logger.info(f"Input SELFIES: {smiles}")
        for token in selfies:
            if token not in stoi.keys():
                raise ValueError("Invalid SELFIES Token: ", token)
        # manual tokens, only start token is added in this case.
        prompt = ["[START]"] + selfies

        return prompt
    elif frag1_smiles is not None and frag2_smiles is not None:
        try:
            frag1_selfie = sf.encoder(frag1_smiles, strict=False)
            frag2_selfie = sf.encoder(frag2_smiles, strict=False)
        except sf.EncoderError as e:
            logger.error("SMILES encode error")
            raise sf.EncoderError(e)

        frag1_symbols = list(sf.split_selfies(frag1_selfie))
        frag2_symbols = list(sf.split_selfies(frag2_selfie))

        prompt = ["[START]"] + frag1_symbols + ["[MASK]"] + frag2_symbols + ["[BRIDGE]"]
        return prompt, frag1_symbols, frag2_symbols
    else:
        if smiles.strip().replace(" ", "") != "":
            logger.info(f"Input SMILES: {smiles}")
            try:
                _ = sf.encoder(smiles)
                test_smiles = smiles
            except sf.EncoderError as e:
                logger.error("SMILES encode error.")
                if alter_prompt:
                    logger.debug("alter_prompt is True, altering prompt.")
                    # start removing characters from end.
                    for i in range(len(smiles), 0, -1):
                        try:
                            test_smiles = smiles[:i]
                            logger.info(f"Altered SMILES: {test_smiles}")
                            _ = sf.encoder(test_smiles)
                            tail = smiles[i:]
                            if len(tail) > 0:
                                logger.info(f"Ingored string: {tail}")
                            break
                        except sf.EncoderError:
                            logger.error("SMILES encode error.")
                            continue
                    raise sf.EncoderError(e)
                else:
                    raise sf.EncoderError(e)

            prompt = ["[START]"] + list(
                sf.split_selfies(sf.encoder(test_smiles, strict=False))
            )
        else:
            logger.debug(
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
        logger.debug(f"min_len is None. Setting min_len to {min_len}")

    if min_len < prompt_len:
        logger.warning(
            f"min_len ({min_len}) < prompt length ({prompt_len}). "
            f"Setting min_len to {prompt_len + 2}"
        )
        min_len = prompt_len + 2

    max_len += prompt_len

    if max_len < min_len:
        max_len += 5
        logger.debug(f"max_len < min_len; Setting max_len to {max_len}")

    logger.info(
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


def output_molecule(
    generation_type, generated_ids, itos, frag1_symbols=[], frag2_symbols=[]
):
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
            if itos[idx] not in ["[END]", "[START]", "[PAD]", "[MASK]", "[BRIDGE]"]
        ]
    )

    if generation_type == "bridge":
        generated_selfies = (
            "".join(frag1_symbols) + generated_selfies + "".join(frag2_symbols)
        )

    logger.info(f"Generated SELFIE: {generated_selfies}")
    generated_smiles = sf.decoder(generated_selfies)
    logger.info(f"Generated SMILES from decoding: {generated_smiles}")
    ignored_token_factor = (
        len(generated_selfies) - len(sf.encoder(generated_smiles))
    ) / len(generated_selfies)
    logger.info(
        f"Proportion of generated tokens ignored by SELFIES decoding : {ignored_token_factor}"
    )

    return generated_smiles, generated_selfies, ignored_token_factor


def generation_loop(
    generation_type,
    model,
    prompt,
    stoi,
    min_len,
    max_len,
    next_atom_criteria,
    temperature,
    k,
    device
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
        if generation_type in ["extend", "evolve"]:
            generated_ids = seed_ids[:]
        elif generation_type == "bridge":
            generated_ids = []
        else:
            raise ValueError("Invalid generation type")

        current_input = torch.tensor([seed_ids]).to(device)

        hidden = None

        for i in range(max_len):
            current_lengths = torch.tensor([current_input.size(1)]).to(device)

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
            elif generation_type == "bridge" and next_atom_id in [
                stoi["[MASK]"],
                stoi["[BRIDGE]"],
            ]:
                last_atom_logits[next_atom_id] = -float("inf")
                next_atom_id = handle_sampling(
                    last_atom_logits, next_atom_criteria, temperature, k
                )

            generated_ids.append(next_atom_id)

            current_input = torch.tensor([[next_atom_id]]).to(device)

    return generated_ids


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
    randomise_prompt=True,
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

    def _generate_from(prompt):
        generated_ids = generation_loop(
            "extend",
            model,
            prompt,
            stoi,
            min_len,
            max_len,
            next_atom_criteria,
            temperature,
            k,
            device
        )
        return generated_ids

    # get default data if model, stoi or itos is not given
    stoi, itos, model = _get_default_data("extend", model, stoi_file, itos_file)

    # put model in evaluation mode
    model.to(device)
    model.eval()

    # check prompt
    prompt = handle_prompt(smiles, selfies, stoi, alter_prompt)
    logger.info(f"Processed prompt: {prompt}")

    # check len
    min_len, max_len = handle_len(prompt, min_len, max_len)

    # generate
    generated_smiles = prompt
    generated_ids = _generate_from(prompt=prompt)
    generated_smiles, generated_selfies, ingored_token_factor = output_molecule(
        "extend", generated_ids, itos
    )

    max_retries = 3
    retry_n = 0

    while generated_smiles == smiles and len(prompt) > 0 and retry_n <= max_retries:
        if alter_prompt:
            prompt = prompt[:-1]
            logger.info(f"Retry {retry_n} with altered prompt : {prompt}")
            generated_ids = _generate_from(prompt=prompt)
            generated_smiles, generated_selfies, ingored_token_factor = output_molecule(
                "extend", generated_ids, itos
            )
            retry_n += 1
        elif randomise_prompt:
            try:
                temp_mol = Chem.MolFromSmiles(smiles)
                logger.info(
                    "Same molecule as input, trying to randomise input smiles."
                )
                prompt = handle_prompt(
                    Chem.MolToSmiles(temp_mol, canonical=False, doRandom=True),
                    selfies,
                    stoi,
                    alter_prompt,
                )
                logger.info(f"Retry {retry_n} with randomised prompt : {prompt}")
                generated_ids = _generate_from(prompt=prompt)
                generated_smiles, generated_selfies, ingored_token_factor = (
                    output_molecule("extend", generated_ids, itos)
                )
                retry_n += 1
            except Exception as e:
                logger.error(f"Randomisation failed : {e}")
                break
        else:
            logger.warning(
                "Same molecule as prompt. This molecule cannot be extended. Try again with a different prompt."
            )
            break

    if ingored_token_factor > 0.1:
        logger.warning("Fraction of ignored tokens due to SELFIES decoding was higher than 0.1")
    m = Chem.MolFromSmiles(generated_smiles)
    if m is None:
        raise ValueError("Invalid molecule")

    return m, generated_smiles, generated_selfies


def evolve(
    model=None,
    stoi_file=None,
    itos_file=None,
    selfies=None,
    smiles="",
    min_len=None,
    max_len=5,
    temperature=1,
    k=10,
    next_atom_criteria="temperature",
    device=device,
    alter_prompt=False,
    n_evolve=4,
):
    """
    Evolve a molecule given a substructure.

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
    if smiles == "":
        _, smiles, _ = extend(smiles=smiles, max_len=max_len)
    generated_mols_list = [Chem.MolFromSmiles(smiles)] + [None] * n_evolve
    generated_smiles_list = [smiles] + [None] * n_evolve
    generated_selfies_list = [sf.encoder(smiles)] + [None] * n_evolve
    current_smiles = smiles
    current_max_len = len(smiles)
    for idx in range(1, n_evolve + 1):
        (
            generated_mols_list[idx],
            generated_smiles_list[idx],
            generated_selfies_list[idx],
        ) = extend(
            model=model,
            stoi_file=stoi_file,
            itos_file=itos_file,
            selfies=selfies,
            smiles=current_smiles,
            min_len=min_len,
            max_len=current_max_len,
            temperature=temperature,
            k=k,
            next_atom_criteria=next_atom_criteria,
            device=device,
            alter_prompt=alter_prompt,
            randomise_prompt=False,
        )
        if current_smiles == generated_smiles_list[idx]:
            logger.warning(
                f"Same molecule detected, early stop at evolution step : {idx}"
            )
            break
        current_smiles = generated_smiles_list[idx]
        current_max_len = len(generated_smiles_list[idx])

    return (
        generated_mols_list[:idx],
        generated_selfies_list[:idx],
        generated_smiles_list[:idx],
    )


def bridge(
    frag1_smiles,
    frag2_smiles,
    model=None,
    stoi_file=None,
    itos_file=None,
    temperature=1,
    k=10,
    next_atom_criteria="temperature",
    device=device
):
    def _generate_from(prompt):
        generated_ids = generation_loop(
            "bridge",
            model=model,
            prompt=prompt,
            stoi=stoi,
            min_len=5,
            max_len=15,
            next_atom_criteria=next_atom_criteria,
            temperature=temperature,
            k=k,
            device=device
        )
        return generated_ids

    stoi, itos, model = _get_default_data("bridge", model, stoi_file, itos_file)

    # put model in evaluation mode
    model.to(device)
    model.eval()

    # first try
    prompt, frag1_symbols, frag2_symbols = handle_prompt(
        frag1_smiles=frag1_smiles, frag2_smiles=frag2_smiles
    )

    generated_ids = _generate_from(prompt)
    generated_smiles, generated_selfies, ignored_token_factor = output_molecule(
        "bridge",
        generated_ids,
        itos,
        frag1_symbols=frag1_symbols,
        frag2_symbols=frag2_symbols,
    )

    max_retries = 3
    retry_n = 0

    while generated_smiles == frag1_smiles and retry_n <= max_retries:
        try:
            temp_mol = Chem.MolFromSmiles(frag1_smiles)
            logger.info("Same molecule as input, trying to randomise input smiles.")
            prompt, frag1_symbols, frag2_symbols = handle_prompt(
                frag1_smiles=Chem.MolToSmiles(temp_mol, canonical=False, doRandom=True),
                frag2_smiles=frag2_smiles,
            )
            logger.info(f"Retry {retry_n} with randomised prompt : {prompt}")
            generated_ids = _generate_from(prompt=prompt)
            generated_smiles, generated_selfies, ingored_token_factor = output_molecule(
                "bridge",
                generated_ids,
                itos,
                frag1_symbols=frag1_symbols,
                frag2_symbols=frag2_symbols,
            )
            retry_n += 1
        except Exception as e:
            logger.error(f"Randomisation failed : {e}")
            break
    m = Chem.MolFromSmiles(generated_smiles)
    if m is None:
        raise ValueError("Invalid molecule")

    return m, generated_smiles, generated_selfies


def decorate(smiles, atom_idx, temperature=1, next_atom_criteria="top_k_temperature"):

    logger.info(f"Input SMILES: {smiles}")
    initial_mol = Chem.MolFromSmiles(smiles)
    if initial_mol is None:
        raise ValueError("Invalid molecule")
    
    if atom_idx>initial_mol.GetNumAtoms():
        raise ValueError(f"Invalid attachment index : attachment index : {atom_idx} is larger than the total number of atoms.")
    
    attachment_atom = initial_mol.GetAtomWithIdx(atom_idx)
    if attachment_atom.GetValence(Chem.ValenceType.EXPLICIT) >= Chem.GetPeriodicTable().GetDefaultValence(attachment_atom.GetSymbol()):
        raise ValueError(f"Atom {atom_idx} ({attachment_atom.GetSymbol()}) is at max valency. Cannot decorate.")

    
    rearranged_mol_smiles = Chem.MolToSmiles(initial_mol,rootedAtAtom=atom_idx,canonical=False)
    rearranged_mol = Chem.MolFromSmiles(rearranged_mol_smiles)
    logger.info(f"Rearranged SMILES: {rearranged_mol_smiles}")
    n_atoms = rearranged_mol.GetNumAtoms()
    logger.info("Redordering atom indices so that selected atom is at the end.")
    new_order = list(range(n_atoms-1,-1,-1))
    logger.info(f"New order: {new_order}")
    mol_reordered = Chem.RenumberAtoms(rearranged_mol, new_order)
    reordered_smiles = Chem.MolToSmiles(mol_reordered, canonical=False)
    logger.info(f"Reordered SMILES: {reordered_smiles}")
    
    logger.info(f"Try extending with extend function, input smiles: {reordered_smiles}")
    m, generated_smiles, generated_selfies = extend(smiles=reordered_smiles,randomise_prompt=False,temperature=temperature,next_atom_criteria=next_atom_criteria)
    
    if Chem.MolToSmiles(mol_reordered,canonical=True) == Chem.MolToSmiles(m,canonical=True):
        # m, generated_smiles, generated_selfies = bridge(frag1_smiles=reordered_smiles,frag2_smiles="C")
        logger.warning("Decoration with full input failed, try with contextual prompting.")
        for radius in [2,1,0]:
            logger.info(f"Try to extract neighbourhood within radius: {radius} of desired attachment atom.")
            if radius == 0:
                logger.warning("Decoration with contextual prompting failed, using only the attachment atom as prompt.")
                context_smiles = initial_mol.GetAtomWithIdx(atom_idx).GetSymbol()
            env = Chem.FindAtomEnvironmentOfRadiusN(mol=initial_mol, radius=radius, rootedAtAtom=atom_idx,enforceSize=True)
            if not env:
                continue
            submol = Chem.PathToSubmol(initial_mol, env, atomMap={})
            context_smiles = Chem.MolToSmiles(submol,kekuleSmiles=False)
            if context_smiles == reordered_smiles:
                logger.warning("Context smiles same as reordered smiles, skipping.")
                continue
            if Chem.MolFromSmiles(context_smiles) is not None:
                logger.info(f"Valid context SMILES found : {context_smiles} at radius : {radius}")
                break
        
        # generate a decorative fragment based on the local environment
        logger.info(f"Try extending with extend function, input smiles: {context_smiles}")
        m1, _, _ = extend(
            smiles=context_smiles, 
            randomise_prompt=True,
            temperature=temperature
        )

        logger.info("Attach decoration to the molecule at desired attachment atom.")
        deco_attachment_indices = list(range(m1.GetNumAtoms())) # possible attachment points in the decoration
        random.shuffle(deco_attachment_indices)
        
        for attach_idx in deco_attachment_indices:
            try:
                combined_mol = RWMol(CombineMols(initial_mol, m1))
                attach_idx_combined_mol = initial_mol.GetNumAtoms() + attach_idx
                logger.info(f"Ataching with atom in decoration. : {attach_idx}, attach index in combined molecule : {attach_idx_combined_mol}")
                combined_mol.AddBond(atom_idx,attach_idx_combined_mol,order=Chem.BondType.SINGLE)
                m2 = combined_mol.GetMol()
                Chem.SanitizeMol(m2)
                generated_smiles = Chem.MolToSmiles(m2)
                logger.info(f"Generated SMILES : {generated_smiles}")
                generated_selfies = sf.encoder(generated_smiles,strict=False)
                return m2, generated_smiles,generated_selfies
            except Exception as e:
                logger.error(f"Failed due to : {e}, trying with a different attachment index.")
        
        raise ValueError("Decoration failed.")
    
    else:
        return m, generated_smiles, generated_selfies