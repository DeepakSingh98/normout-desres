import torch
import torch.nn as nn

from abc import ABC

class Custom_Layer(ABC):

    def set_index(self, num_id):
        self.num_id = num_id

    def forward(self):
        pass