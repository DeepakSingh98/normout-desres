import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms

class NormOutModel(pl.LightningModule):
    def __init__(self, normout_fc1=False, normout_fc2=False, optimizer="SGDM", lr=0.001, batch_size=64, num_workers=4, **kwargs):
        super(NormOutModel, self).__init__()

        # model
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # settings
        self.normout_fc1 = normout_fc1
        self.normout_fc2 = normout_fc2
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        # trackers
        self.fc1_neuron_tracker = torch.zeros(self.fc1.out_features)
        self.fc2_neuron_tracker = torch.zeros(self.fc2.out_features)


        # dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.training_set = torchvision.datasets.FashionMNIST(
            "./data", train=True, transform=transform, download=True
        )
        self.validation_set = torchvision.datasets.FashionMNIST(
            "./data", train=False, transform=transform, download=True
        )
        

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

    def configure_optimizers(self):
        if self.optimizer == "SGDM":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer} not implemented")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, run_info = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log({
            "Train Loss": loss, 
            "Train Accuracy": F.accuracy(y_hat, y), 
            "FC1 Average Percent Activated": run_info["fc1_mask"].sum(dim=0).mean(),
            "FC2 Average Percent Activated": run_info["fc2_mask"].sum(dim=0).mean()
            },
        )
        self.fc1_neuron_tracker += run_info["fc1_mask"].sum(dim=0)
        self.fc2_neuron_tracker += run_info["fc2_mask"].sum(dim=0)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log({"Validation Loss": loss, "Validation Accuracy": F.accuracy(y_hat, y)})
        return loss

    def on_train_epoch_start(self) -> None:
        self.fc1_neuron_tracker.zero_()
        self.fc2_neuron_tracker.zero_()
    
    def on_train_epoch_end(self) -> None:
        self.log({"FC1 Dead Neuron Prevalence": (self.fc1_neuron_tracker == 0).sum().item() / self.fc1_neuron_tracker.numel()})
        self.log({"FC2 Dead Neuron Prevalence": (self.fc2_neuron_tracker == 0).sum().item() / self.fc2_neuron_tracker.numel()})

    # dataloaders
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )