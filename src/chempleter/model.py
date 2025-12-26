import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class ChempleterModel(nn.Module):
    """
    A PyTorch GRU model for generating SELFIES tokens

    :param vocab_size: Size of the vocabulary
    :type vocab_size: int
    :param embedding_dim: Dimension of the embedding layer, defaults to 256
    :type embedding_dim: int
    :param hidden_dim: Dimension of the hidden state, defaults to 512
    :type hidden_dim: int
    :param num_layers: Number of GRU layers, defaults to 2
    :type num_layers: int
    """

    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, tensor_lengths, hidden=None):
        """
        Forward pass of the model.

        :param x: Input tensor of token indices
        :type x: torch.Tensor
        :param tensor_lengths: Lengths of sequences in the batch
        :type tensor_lengths: torch.Tensor
        :param hidden: Initial hidden state, defaults to None
        :type hidden: torch.Tensor or None
        :return: Model logits and final hidden state
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(
            embedded, tensor_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, hidden = self.gru(packed_embedded, hidden)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)
        logits = self.fc(out)

        return logits, hidden
