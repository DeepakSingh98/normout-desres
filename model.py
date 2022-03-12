import torch
import torch.nn as nn
import torch.nn.functional as F


class NormOutModel(nn.Module):
    def __init__(self, normout_fc1=False, normout_fc2=False):
        super(NormOutModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.normout_fc1 = normout_fc1
        self.normout_fc2 = normout_fc2

    def forward(self, x):
        run_info = dict()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        if self.normout_fc1:
            # divide by biggest value in the activation per input
            norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        run_info["fc1_mask"] = x > 0
        x = F.relu(self.fc2(x))
        if self.normout_fc2:
            # divide by biggest value in the activation per input
            norm_x = x / torch.max(x, dim=1, keepdim=True)[0]
            x_mask = torch.rand_like(x) < norm_x
            x = x * x_mask
        run_info["fc2_mask"] = x > 0
        x = self.fc3(x)
        return x, run_info

