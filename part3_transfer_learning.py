# -----------------------------
# Importing all necessary libraries
# -----------------------------

import os  # For handling file paths and directories
import numpy as np  # For numerical operations like random seed setting
import pandas as pd  # Useful for working with CSV files (e.g., Kaggle submission)
import urllib.request  # Can be used to download resources if needed
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network building blocks (e.g., layers)
import torch.optim as optim  # Optimization algorithms like AdamW
import torchvision  # For loading datasets and pretrained models
import torchvision.transforms as transforms  # For image preprocessing and data augmentation
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset  # Data handling utilities
from tqdm.auto import tqdm  # Progress bar for training and validation loops
import wandb  # Weights & Biases for experiment tracking


# defining a transfer learning model using ResNet50


class ResNet50Transfer(nn.Module):
    def __init__(self):
        super(ResNet50Transfer, self).__init__()  # Initialize the parent class
        self.model = torchvision.models.resnet50(pretrained=True)  # Load pretrained ResNet50 from ImageNet
        self.model.fc = nn.Sequential(  # replacing the final fully connected layer
            nn.Dropout(0.2),  # adding dropout for regularization
            nn.Linear(self.model.fc.in_features, 100)  # mapping 2048 features to 100 CIFAR-100 classes
        )

    def forward(self, x):
        return self.model(x)  # Pass input through the model

# Training function for one epoch
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]  # Set the computation device (CPU, CUDA, MPS)
    model.train()  # Set model to training mode (enables dropout and batchnorm)
    running_loss = 0.0  # Cumulative loss across batches
    correct = 0  # Number of correctly predicted samples
    total = 0  # Total number of samples seen

    # Progress bar over training data
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # computing loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss
        _, predicted = outputs.max(1)  # Get predicted class
        total += labels.size(0)  # Total number of labels in batch
        correct += predicted.eq(labels).sum().item()  # Count correct predictions

        # Update progress bar with metrics
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    return running_loss / len(trainloader), 100. * correct / total  # Return average loss and accuracy

# Validation function (no gradient updates)

def validate(model, valloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    return running_loss / len(valloader), 100. * correct / total

# Main training script


def main():
    CONFIG = {
        "model": "ResNet50",  # Name of model architecture
        "batch_size": 64,  # Batch size
        "learning_rate": 0.0005,  # Initial learning rate
        "epochs": 30,  # Number of training epochs
        "num_workers": 4,  # For DataLoader parallelism
        "device": "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu",
        "data_dir": "./data",  # CIFAR-100 data path
        "ood_dir": "./data/ood-test",  # OOD test data path
        "wandb_project": "sp25-ds542-challenge",  # WandB project name
        "seed": 42  # Random seed for reproducibility
    }

    # Set all random seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    import random
    random.seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed(CONFIG["seed"])

    # Pretty print config
    print("\nCONFIG Dictionary:")
    import pprint
    pprint.pprint(CONFIG)

    # -----------------------------
    # Data Preprocessing & Augmentation
    # -----------------------------
    IMAGENET_MEAN = (0.485, 0.456, 0.406)  # ResNet50 expects ImageNet normalization
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),  # Horizontal flip
        transforms.RandomRotation(15),  # Random rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # Normalize to ImageNet stats
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # -----------------------------
    # Dataset Loading
    # -----------------------------
    full_train = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(full_train))  # 80% for training
    val_size = len(full_train) - train_size  # 20% for validation
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(CONFIG["seed"]))

    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, transform=transform_test)

    trainset = Subset(trainset, train_subset.indices)  # Create training subset
    valset = Subset(valset, val_subset.indices)  # Create validation subset

    # -----------------------------
    # Data Loaders
    # -----------------------------
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # -----------------------------
    # Model, Loss, Optimizer, Scheduler
    # -----------------------------
    model = ResNet50Transfer().to(CONFIG["device"])  # Initialize model and move to device

    print("\nModel summary:\n", model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Loss with label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)  # AdamW optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])  # Learning rate scheduler

    # -----------------------------
    # Initialize Weights & Biases
    # -----------------------------
    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)
    wandb.watch(model)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()  # Close WandB run

    # Final Evaluation
    import eval_cifar100  # Provided eval script for clean CIFAR-100
    import eval_ood  # Provided eval script for OOD data

    model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))
    model.eval()

    # evaluating on CIFAR-100 test set
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # evaluating on OOD test set and save predictions
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv("submission_ood_transfer_resnet50.csv", index=False)
    print("submission_ood_transfer_resnet50.csv created successfully.")

if __name__ == '__main__':
    main()  # executing main function
