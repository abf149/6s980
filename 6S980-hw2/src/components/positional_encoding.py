import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        
        # The frequency multiplier should start from 2 * pi and double for each octave.
        # We'll have two frequencies for each octave (one for sin and one for cos), hence the "* 2".
        self.num_octaves = num_octaves
        self.num_frequencies = num_octaves * 2

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        # Find out how many samples we have and their dimensionality.
        did_unsqueeze=False
        if len(samples.shape)==1:
            did_unsqueeze=True
            samples=samples.unsqueeze(0)
        #num_samples, dim = samples.shape
        
        # Initiate a list to store the embeddings.
        embeddings = []
        
        # Go through each frequency (octave).
        for freq_power in range(self.num_octaves):
            # Calculate the frequency multiplier: k = 2^octave
            k = 2 ** freq_power
            # Sinusoidal and Cosinusoidal encoding
            sin_embedding = torch.sin(2 * torch.pi * k * samples)
            cos_embedding = torch.cos(2 * torch.pi * k * samples)
            embeddings.extend([sin_embedding, cos_embedding])

        # Stack the embeddings together.
        res=torch.cat(embeddings, dim=1)
        if did_unsqueeze:
            return res.squeeze(0)
        else:
            return res

    def d_out(self, dimensionality: int) -> int:
        # This function returns the output dimensionality of the positional encoding
        return self.num_frequencies * dimensionality