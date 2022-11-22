import torch
from torch import nn
import torch.nn.functional as F
import math

class Embedding(nn.Module):
    """Embedding layer."""
    def __init__(self, d_model, vocab) -> None:

        # Initialize the weights with maximum vocab and the dimension ouf the weight output matrix
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """Return the weight matrix.

        Args:
            x: input vector
        """

        # Paper: Multiply the weights with square root of d_model
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    