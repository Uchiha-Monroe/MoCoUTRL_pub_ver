import nntplib
from turtle import forward
import torch
import torch.nn as nn



class LinearMicroClassfier(nn.Module):
    def __init__(self, num_dims: int=768, num_class: int=134):
        super().__init__()
        
        self.fc1 = nn.Linear(num_dims, 4*num_dims)
        self.fc2 = nn.Linear(4*num_dims, num_dims)
        self.fc3 = nn.Linear(num_dims, num_class)
        
        self.gelu = nn.GELU()
        # self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return x



class LinearMacroClassfier(nn.Module):
    def __init__(self, num_dims: int=768, num_class: int=7):
        super().__init__()
        
        self.fc1 = nn.Linear(num_dims, 4*num_dims)
        self.fc2 = nn.Linear(4*num_dims, num_dims)
        self.fc3 = nn.Linear(num_dims, num_class)
        
        self.gelu = nn.GELU()
        # self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

