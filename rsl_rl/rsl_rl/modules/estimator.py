import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical

class LSTMEncoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
      
        # self.output_mlp = nn.Sequential(
        #                         nn.Linear(256, 32),
        #                         nn.ELU()
        #                     )

        encoder_dims = [256,128]
        self.output_dim = 8 + 16
        mlp_layers = []
        mlp_layers.append(nn.Linear(256,encoder_dims[0]))
        mlp_layers.append(nn.ELU())
        for l in range(len(encoder_dims)):
           if l == len(encoder_dims) -1:
              mlp_layers.append(nn.Linear(encoder_dims[l],self.output_dim))
           else:
              mlp_layers.append(nn.Linear(encoder_dims[l],encoder_dims[l+1]))
              mlp_layers.append(nn.ELU())
        self.output_mlp = nn.Sequential(*mlp_layers)

              

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.h = None 
        self.c = None 
    
    def forward(self,obs):
        batch_size = obs.shape[0]
        if self.h is None or self.c is None:
            self.h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=obs.device)
            self.c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=obs.device)
        scan_latent, (self.h,self.c) = self.rnn(obs[:, None, :], (self.h,self.c))
        scan_latent = self.output_mlp(scan_latent.squeeze(1))
        return scan_latent
    
    def detach_hidden_states(self):
        if self.h is not None:
            self.h = self.h.detach().clone()
        if self.c is not None:
            self.c = self.c.detach().clone()
        