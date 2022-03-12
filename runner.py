import argparse
import datetime
from model import NormOutModel
from dataloader import training_set, validation_set
import torch

from datetime import datetime
import wandb


# accept command line arguments for epochs, batch size, number of workers, normout_fc1, normout_fc2
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--normout-fc1", action="store_true", default=False)
parser.add_argument("--normout-fc2", action="store_true", default=False)
args = parser.parse_args()

model = NormOutModel(normout_fc1=args.normout_fc1, normout_fc2=args.normout_fc2)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)


def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0
    num_correct = 0
    num_total = 0
    num_active_fc1 = 0
    total_fc1_activations = 0
    fc1_neuron_tracker = torch.zeros(model.fc1.out_features)
    num_active_fc2 = 0
    total_fc2_activations = 0
    fc2_neuron_tracker = torch.zeros(model.fc2.out_features)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, run_info = model(inputs)
        num_active_fc1 += run_info["fc1_mask"].sum().item()
        total_fc1_activations += run_info["fc1_mask"].numel()
        fc1_neuron_tracker += run_info["fc1_mask"].sum(dim=0)
        num_active_fc2 += run_info["fc2_mask"].sum().item()
        total_fc2_activations += run_info["fc2_mask"].numel()
        fc2_neuron_tracker += run_info["fc2_mask"].sum(dim=0)

        # log number of correct predictions
        num_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
        num_total += len(labels)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % len(training_loader) == len(training_loader) - 1:
            last_loss = running_loss / len(training_loader)  # loss per batch
            wandb.log(
                {
                    "Train Loss": last_loss,
                    "FC1 Average Percent Activated": num_active_fc1
                    / total_fc1_activations,
                    "FC2 Average Percent Activated": num_active_fc2
                    / total_fc2_activations,
                    "PercentDeadFC1": torch.sum(fc1_neuron_tracker == 0).item()
                    / fc1_neuron_tracker.shape[0],
                    "PercentDeadFC2": torch.sum(fc2_neuron_tracker == 0).item()
                    / fc2_neuron_tracker.shape[0],
                    "Train Accuracy": num_correct / num_total,
                }
            )
            running_loss = 0.0
            print(
                "  batch {} loss: {} acc: {}".format(
                    i + 1, last_loss, num_correct / num_total
                )
            )
            # write accuracy to tensorboard

    return last_loss, num_correct / num_total


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
epoch_number = 0
tags = []
if model.normout_fc1:
    tags.append("normout_fc1")
if model.normout_fc2:
    tags.append("normout_fc2")
wandb.init(
    project="normout",
    name=(("-").join(tags) + "-" + timestamp) if len(tags) > 0 else timestamp,
    tags=tags,
)

best_vloss = 1000000.0

for epoch in range(args.epochs):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_acc = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    num_correct = 0
    num_total = 0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs, run_info = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
        num_correct += torch.sum(torch.argmax(voutputs, dim=1) == vlabels).item()
        num_total += len(vlabels)

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = num_correct / num_total
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
    print("ACC train {} valid {}".format(avg_acc, avg_vacc))

    # Log the running loss averaged per batch
    # for both training and validation
    wandb.log(
        {"Validation Loss": avg_vloss, "Validation Accuracy": avg_vacc,}
    )

    # Save the model if it's the best we've seen so far
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), "best_model.pt")

    epoch_number += 1

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "model_{}_{}".format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

