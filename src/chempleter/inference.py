import json
import torch
import selfies as sf
from rdkit import Chem
import random

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def handle_prompt(smiles,selfies,stoi,alter_prompt):
    if selfies is not None:
        print(f"Input SELFIES: {smiles}")
        for i in selfies:
            if i not in stoi.keys():
                raise ValueError("Invalid Token.")
        prompt = ["[START]"]+selfies
    else:
        if smiles.strip().replace(" ","")!="":
            print(f"Input SMILES: {smiles}")

            try:
                input_selfies = sf.encoder(smiles)
                test_smiles = smiles
            except sf.EncoderError as e:
                if alter_prompt:
                    for i in range(len(smiles)-1, 0, -1):
                        try:
                            test_smiles = smiles[:i]
                            input_selfies = sf.encoder(test_smiles)
                            tail = smiles[i:]
                            if len(tail)>0:
                                print(f"Ingored string: {tail}")
                            break
                        except sf.EncoderError:
                            continue
                else:
                    raise sf.EncoderError(e)

            prompt=["[START]"]+list(sf.split_selfies(sf.encoder(test_smiles,strict=False)))
        else:
            prompt=["[START]"]

    return prompt

def handle_sampling(last_atom_logits,next_atom_criteria,temperature,k):
    if next_atom_criteria == "random":
        next_atom_criteria = random.choice(["greedy","temperature","top_k_temperature"])

    if next_atom_criteria=="greedy":
        next_atom_id = torch.argmax(last_atom_logits).item()
    elif next_atom_criteria=="temperature":
        probs = torch.softmax(last_atom_logits / temperature, dim=-1)
        next_atom_id = torch.multinomial(probs, 1).item()
    elif next_atom_criteria=="top_k_temperature":
        topk_probs, topk_indices = torch.topk(torch.nn.functional.softmax(last_atom_logits/temperature, dim=-1), k)
        topk_probs /= topk_probs.sum()
        next_atom_id = topk_indices[torch.multinomial(topk_probs, 1)].item()
    else:
        print("Using default sampling.")
        next_atom_id = torch.argmax(last_atom_logits).item()

    return next_atom_id

def generation_loop(model,prompt,stoi,min_len,max_len,next_atom_criteria, temperature, k):

    with torch.no_grad():

        seed_ids = [stoi[symbol] for symbol in prompt]
        generated_ids = seed_ids[:]
        current_input = torch.tensor([seed_ids]).to(device)
        hidden=None

        for i in range(max_len):
            logits, hidden = model(current_input, hidden)
            last_atom_logits = logits[0, -1, :]

            next_atom_id = handle_sampling(last_atom_logits,next_atom_criteria,temperature,k)
            
            generated_ids.append(next_atom_id)

            # if next_atom_id == stoi['[END]']:
            #     break

            if next_atom_id == stoi['[END]']:
                if len(generated_ids) < min_len:
                    # Mask out END token and resample
                    last_atom_logits[stoi['[END]']] = -float('inf')
                    next_atom_id = handle_sampling(last_atom_logits, next_atom_criteria, temperature, k)
                    generated_ids.append(next_atom_id)
                else:
                    break
            else:
                generated_ids.append(next_atom_id)
            
            current_input = torch.tensor([[next_atom_id]]).to(device)
    
    return generated_ids


def handle_len(prompt,min_len,max_len):
    prompt_len = len(prompt)
    
    if min_len is None:
        min_len = prompt_len + 2
    
    if min_len < prompt_len:
        print(f"Warning: min_len ({min_len}) < prompt length ({prompt_len}). "
              f"Setting min_len to {prompt_len + 2}")
        min_len = prompt_len + 2
    
    max_len += prompt_len

    if max_len < min_len:
        max_len+5
    
    # if max_len < prompt_len:
    #     raise ValueError(f"max_len ({max_len}) must be >= prompt length ({prompt_len})")
    
    return min_len,max_len
    

def extend(model,stoi_file,itos_file,selfies=None,smiles="",min_len=None,max_len=50,temperature=0.7,k=10,next_atom_criteria="top_k_temperature",device=device,alter_prompt=False):

    model.to(device)
    model.eval() # put model in evaluation mode

    with open(stoi_file) as f:
        stoi = json.load(f)
    with open(itos_file) as f:
        itos = json.load(f)

    prompt = handle_prompt(smiles,selfies,stoi,alter_prompt)
    
    print(f"Input prompt: {prompt}")

    min_len,max_len = handle_len(prompt,min_len,max_len)
    print(min_len,max_len)
    
    generated_smiles = prompt
    
    generated_ids = generation_loop(model,prompt,stoi,min_len,max_len,next_atom_criteria, temperature, k)
    generated_smiles,generated_selfies = output_molecule(generated_ids,itos)

    retry_n = 0
    while generated_smiles==smiles and len(prompt)>0:
        if alter_prompt:
            prompt = prompt[:-1]
            retry_n+=1
            print(f"Retry {retry_n} with altered prompt : {prompt}")
            generated_ids = generation_loop(model,prompt,stoi,min_len,max_len,next_atom_criteria, temperature, k)
            generated_smiles,generated_selfies = output_molecule(generated_ids,itos)
        else:
            print("Same molecule as prompt. This molecule cannot be extended. Try again with a different prompt.")
            break

    m = Chem.MolFromSmiles(generated_smiles)
    if m is None:
        raise ValueError("Invalid molecule")
    
    return m,generated_smiles,generated_selfies

def output_molecule(generated_ids,itos):
    generated_selfies = "".join([itos[idx] for idx in generated_ids if itos[idx] not in ['[END]','[START]','[PAD]']])
    print(f"Generated SELFIE: {generated_selfies}")
    generated_smiles = sf.decoder(generated_selfies)
    print(f"Generated SMILES: {generated_smiles}")

    return generated_smiles,generated_selfies
