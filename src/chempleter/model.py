import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class ChempleterModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim=128,hidden_dim=256,num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim,padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,vocab_size)

    def forward(self,x, hidden=None):
        embedded = self.embedding(x)
        out,hidden = self.gru(embedded,hidden)
        logits = self.fc(out)

        return logits,hidden