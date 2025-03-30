
import torch  # PyTorch base library for tensor operations and GPU support
import torch.nn as nn  # Module to build neural network layers
import torch.nn.functional as F  # Functional interface for layers/activations like relu, softmax
import torch.optim as optim  # Optimization algorithms such as AdamW

import torchvision  # For loading popular datasets and pretrained models
import torchvision.transforms as transforms  # Image preprocessing and augmentation utilities

import os  # File and path handling
import numpy as np  # scientific computing and numerical operations
import pandas as pd  # data manipulation and saving outputs to CSV

from tqdm.auto import tqdm  # for clean progress bars in training and validation loops
import wandb  # Weights & Biases for experiment tracking, logging, and model versioning
import json  # To load or dump configuration files if needed
from torch.utils.data import random_split, DataLoader  # Dataset splitting and efficient batch loading
from torchvision.models import resnet18  # Import ResNet18 model architecture from torchvision

# define Sophisticated CNN using ResNet18 as base ===
class SophisticatedCNN(nn.Module):
    def __init__(self):
        super(SophisticatedCNN, self).__init__()
        # Load standard ResNet18 with output layer for 100 classes (CIFAR-100)
        base_model = resnet18(num_classes=100)
        num_ftrs = base_model.fc.in_features

        # Replace the final classification layer to include dropout for regularization
        base_model.fc = nn.Sequential(
            nn.Dropout(p=0.2),              # Dropout to reduce overfitting
            nn.Linear(num_ftrs, 100)        # Fully connected layer for 100 output classes
        )

        self.model = base_model  # assigning modified model

    def forward(self, x):
        return self.model(x)  #  forward pass through the network

#Training Function 
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()  # Enable training mode (which is important for Dropout and BatchNorm)

    running_loss = 0.0
    correct = 0
    total = 0

    # Display progress per batch using tqdm
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU or CPU
        optimizer.zero_grad()  # Zero the gradients before each step
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Get predictions and calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

#Validation Function 
def validate(model, valloader, criterion, device):
    model.eval()  # evaluation mode: disables dropout and batch norm updates
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disabling gradient calculation to save memory/speed up
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute validation loss

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# main Function: Full Training Pipeline ===
def main():
    # Configuration dictionary holding all key hyperparameters
    CONFIG = {
        "model": "SophisticatedCNN",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 60,
        "num_workers": 4,
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
        "best_model_path": "best_model.pth"
    }

    # Print configuration details
    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # data Augmentation & Normalization for training and validation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip images
        transforms.RandomRotation(15),  # Small angle rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color distortion
        transforms.RandomCrop(32, padding=4),  # Crop image with padding
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),  # CIFAR-100 normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),  # No augmentation for test/val
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    # Load and split CIFAR-100 dataset
    full_train = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(full_train))  # 80% training
    val_size = len(full_train) - train_size  # 20% validation
    trainset, valset = random_split(full_train, [train_size, val_size])  # Perform split

    # creating data loaders for training, validation, and test
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # initializing model and move to GPU/CPU
    model = SophisticatedCNN().to(CONFIG["device"])
    print("\nModel summary:")
    print(model)

    # Loss function with no label smoothing (CrossEntropyLoss is standard for classification)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)  # Regularized Adam
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])  # Smoothly reduce LR

    # Initialize wandb run
    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)
    wandb.watch(model)

    # Training loop with model checkpointing
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()  # Adjust learning rate per epoch

        # logging all metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # saving best performing model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["best_model_path"])  # saving model weights
            wandb.save(CONFIG["best_model_path"])

    wandb.finish()  # Close wandb session

    # === Evaluation ===
    import eval_cifar100  # Provided helper for test set accuracy
    import eval_ood  # Provided helper for OOD evaluation

    model.load_state_dict(torch.load(CONFIG["best_model_path"]))
    model.eval()

    # CIFAR-100 Test Accuracy
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # OOD evaluation
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood_sophisticated.csv", index=False)
    print("submission_ood_sophisticated.csv created successfully.")

# starting point for data pipeline
if __name__ == '__main__':
    main()
