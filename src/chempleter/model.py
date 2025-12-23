import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class ChempleterModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim=256,hidden_dim=512,num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,vocab_size)

    def forward(self,x, tensor_lengths, hidden=None):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, tensor_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, hidden = self.gru(packed_embedded, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,padding_value=0)
        logits = self.fc(out)

        return logits,hidden