import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch

def set_tags(args):
    """
    Set wandb tags for the experiment.
    """
    tags = [args.model_name, args.optimizer, args.dataset_name]
    if args.custom_layer_name is not None:
        tags.append(args.custom_layer_name)
        if args.custom_layer_name == "Dropout":
            tags.append(f"p_{str(args.dropout_p)}")
        elif args.custom_layer_name == "TopK":
            tags.append(f"k_{str(args.topk_k)}")
    if args.custom_tag:
        tags.append(args.custom_tag)
    if args.pretrained:
        tags.append("pretrained")
    if not args.no_data_augmentation:
        tags.append("DataAug")
    if not args.no_abs and (args.custom_layer_name == "NormOut" or args.custom_layer_name == "SigmoidOut"):
        tags.append(f'use_abs')
    if args.custom_layer_name == "NormOut":
        tags.append(f"{args.normalization_type}")
    #if (args.custom_layer_name == "NormOut" or args.custom_layer_name == "Dropout") and args.on_at_inference:
     #   tags.append("on_at_inference")
    if args.custom_layer_name is not None and args.replace_layers is not None:
        tags.append(f"replaced_{'_'.join([str(i) for i in args.replace_layers])}")
    if args.custom_layer_name is not None and args.insert_layers is not None:
        tags.append(f"inserted_{'_'.join([str(i) for i in args.insert_layers])}")
    if args.remove_layers is not None:
        tags.append(f"removed_{'_'.join([str(i) for i in args.remove_layers])}")
    if args.custom_layer_name == "SigmoidOut":
        tags.append(f"{args.normalization_type}")
    tags.append(f"Seed={args.seed}")
    return tags

# Credit: https://github.com/GMvandeVen/continual-learning/blob/master/data.py

#----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].
    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indices = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indices.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample

# Continual learning utils

class Task_Scheduler():

    def __init__(self, labels_per_task: list):

        self.labels_per_task = labels_per_task

        for i, labels in enumerate(labels_per_task):
            print(f"Task {i+1}: classes {labels}")
    
    def start_task(self, current_task):
        print(f"Starting Task {current_task+1} (classes: {self.labels_per_task[current_task]})")
    
    def end_task(self, current_task):
        print(f"End of Task {current_task+1} (classes: {self.labels_per_task[current_task]})")
        current_task += 1
        return current_task
