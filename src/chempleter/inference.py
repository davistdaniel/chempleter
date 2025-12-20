import json
import torch
import selfies as sf
from rdkit import Chem

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def extend(model,stoi_file,itos_file,smiles="",min_len=None,max_len=50,temperature=0.5,k=10,next_atom_criteria="multinomial",device=device):

    model.to(device)
    
    with open(stoi_file) as f:
        stoi = json.load(f)
    with open(itos_file) as f:
        itos = json.load(f)

    model.eval() # put model in evaluation mode

    
    if smiles.strip().replace(" ","")!="":
        print(f"Input SMILES: {smiles}")
        prompt=["[START]"]+list(sf.split_selfies(sf.encoder(smiles,strict=False)))
    else:
        prompt=["[START]"]
    
    if min_len is None:
         min_len = int(len(prompt)+2)

    with torch.no_grad():
          
        seed_ids = [stoi[symbol] for symbol in prompt]
        generated_ids = seed_ids[:]
        current_input = torch.tensor([seed_ids]).to(device)
        hidden=None

        for i in range(max_len):
            logits, hidden = model(current_input, hidden)
            last_atom_logits = logits[0, -1, :]

            if next_atom_criteria=="argmax":
                next_atom_id = torch.argmax(last_atom_logits).item()
            else:
                topk_probs, topk_indices = torch.topk(torch.nn.functional.softmax(last_atom_logits/temperature, dim=-1), k)
                topk_probs /= topk_probs.sum()
                next_atom_id = topk_indices[torch.multinomial(topk_probs, 1)].item()
                 
            generated_ids.append(next_atom_id)

            if next_atom_id == stoi['[END]']:
                if len(generated_ids)<min_len:
                      last_atom_logits[stoi['[END]']] = -float('inf')
                else:
                    break
            
            current_input = torch.tensor([[next_atom_id]]).to(device)
    generated_selfies = "".join([itos[idx] for idx in generated_ids[1:-1]])
    print(f"Generated SELFIE: {generated_selfies}")
    generated_smiles = sf.decoder(generated_selfies)
    print(f"Generated SMILES: {generated_smiles}")

    m = Chem.MolFromSmiles(generated_smiles)

    return m