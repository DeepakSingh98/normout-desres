import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST(
    "./data", train=True, transform=transform, download=True
)
validation_set = torchvision.datasets.FashionMNIST(
    "./data", train=False, transform=transform, download=True
)

# Class labels
classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)

# Report split sizes
print("Training set has {} instances".format(len(training_set)))
print("Validation set has {} instances".format(len(validation_set)))

# if __name__ == "__main__":
    # Create data loaders for our datasets; shuffle for training, not for validation
    # training_loader = torch.utils.data.DataLoader(
    #     training_set, batch_size=4, shuffle=True, 
    # )
    # validation_loader = torch.utils.data.DataLoader(
    #     validation_set, batch_size=4, shuffle=False,
    # )
#     # VERIFICATION
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Helper function for inline image display
#     def matplotlib_imshow(img, one_channel=True):
#         if one_channel:
#             img = img.mean(dim=0)
#         img = img / 2 + 0.5  # unnormalize
#         npimg = img.numpy()
#         plt.imshow(npimg, cmap="Greys")
#         plt.show()
#         # else:
#         #     plt.imshow(np.transpose(npimg, (1, 2, 0)))

#     dataiter = iter(training_loader)
#     images, labels = dataiter.next()

#     # Create a grid from the images and show them
#     img_grid = torchvision.utils.make_grid(images)
#     matplotlib_imshow(img_grid, one_channel=True)
#     print("  ".join(classes[labels[j]] for j in range(4)))
