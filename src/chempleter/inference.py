import json
import torch
import selfies as sf
from rdkit import Chem

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def handle_prompt(smiles,selfies,stoi):
    if selfies is not None:
        print(f"Input SELFIES: {smiles}")
        for i in selfies:
            if i not in stoi.keys():
                raise ValueError("Invalid Token.")
        prompt = ["[START]"]+selfies
    else:
        if smiles.strip().replace(" ","")!="":
            print(f"Input SMILES: {smiles}")

            for i in range(len(smiles), 0, -1):
                try:
                    test_smiles = smiles[:i]
                    input_selfies = sf.encoder(test_smiles)
                    tail = smiles[i:]
                    if len(tail)>0:
                        print(f"Ingored string: {tail}")
                    break
                except sf.EncoderError:
                    continue

            prompt=["[START]"]+list(sf.split_selfies(sf.encoder(test_smiles,strict=False)))
        else:
            prompt=["[START]"]

    return prompt

def extend(model,stoi_file,itos_file,selfies=None,smiles="",min_len=None,max_len=50,temperature=0.7,k=10,next_atom_criteria="top_k_temperature",device=device):

    model.to(device)
    model.eval() # put model in evaluation mode

    with open(stoi_file) as f:
        stoi = json.load(f)
    with open(itos_file) as f:
        itos = json.load(f)

    prompt = handle_prompt(smiles,selfies,stoi)
    print(f"Input prompt: {prompt}")
    
    if min_len is None or min_len<len(prompt):
         min_len = int(len(prompt)+2)
    
    with torch.no_grad():

        seed_ids = [stoi[symbol] for symbol in prompt]
        generated_ids = seed_ids[:]
        current_input = torch.tensor([seed_ids]).to(device)
        hidden=None

        for i in range(max_len):
            logits, hidden = model(current_input, hidden)
            last_atom_logits = logits[0, -1, :]

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
            
            generated_ids.append(next_atom_id)

            if next_atom_id == stoi['[END]']:
                if len(generated_ids)<min_len:
                      last_atom_logits[stoi['[END]']] = -float('inf')
                else:
                    break
            
            current_input = torch.tensor([[next_atom_id]]).to(device)
    
    generated_selfies = "".join([itos[idx] for idx in generated_ids if itos[idx] not in ['[END]','[START]','[PAD]']])
    print(f"Generated SELFIE: {generated_selfies}")
    generated_smiles = sf.decoder(generated_selfies)
    print(f"Generated SMILES: {generated_smiles}")

    m = Chem.MolFromSmiles(generated_smiles)
    if m is None:
        raise ValueError("Invalid molecule")
    
    return m,generated_smiles,generated_selfies