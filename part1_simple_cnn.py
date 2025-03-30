# PyTorch libraries for model building, training, and optimization
import torch 
import torch.nn as nn  # Contains all building blocks for neural networks (layers and loss functions)
import torch.nn.functional as F  # Functional interface to PyTorch layers (e.g., activation functions like F.relu)
import torch.optim as optim  # Contains standard optimization algorithms like SGD, Adam, etc.
import torchvision  # loading datasets and pretrained models
import torchvision.transforms as transforms  # Tools for preprocessing and augmenting image data
# built-in Python libraries for file handling and numerical operations
import os  # For file path and OS-level operations (e.g., checking directories)
import numpy as np  # Numerical computing library, useful for handling arrays, statistics, etc.
import pandas as pd  # Powerful library for structured data manipulation (used for saving predictions to CSV)
# tqdm is used for displaying dynamic progress bars during training and validation
from tqdm.auto import tqdm  # `auto` picks the best available display method (e.g., notebook or terminal)
# weights & Biases (wandb) is a tool for tracking experiments, logging metrics, visualizing model training
import wandb  # Logs metrics, saves models, allows comparing results across runs visually on wandb.ai
import json  # Useful for saving/loading configs (e.g., hyperparameters, results)
# PyTorch utilities for loading and batching datasets efficiently
from torch.utils.data import random_split, DataLoader  # `random_split` for train/val split, `DataLoader` for batching and shuffling


# Define a simple CNN model architecture for CIFAR-100 classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Feature extractor: series of Conv → ReLU → MaxPool layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input: 3x32x32 → output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # output: 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # output: 128x8x8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128x4x4

            nn.Flatten(),  # Flatten to vector of length 128 * 4 * 4
            nn.Dropout(0.3)  # Regularization to prevent overfitting
        )

        # Classifier: Fully connected layers to map features → class scores
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # Output layer: 100 classes in CIFAR-100
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# train function for one epoch
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()  # Enable dropout, batchnorm, etc.

    running_loss = 0.0  # Accumulate total loss
    correct = 0  # Count correct predictions
    total = 0  # Total number of examples seen

    # progress bar for visual tracking of training
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # Iterating through all batches in the training set
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients from previous step
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

        # updating running loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update tqdm bar with current metrics
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    # Calculate average loss and accuracy for the epoch
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# validation loop 
def validate(model, valloader, criterion, device):
    model.eval()  # Turn off dropout, batchnorm, etc.
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed during validation
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

    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# Main function: Entry point to training, validation, and evaluation
def main():
    # setting up all training parameters in a single dictionary
    CONFIG = {
        "model": "SimpleCNN",  # Track model name
        "batch_size": 64,
        "learning_rate": 0.005,
        "epochs": 50,
        "num_workers": 4,  # Use multiple workers to load data in parallel
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Location of dataset
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42
    }

    # Display the configuration
    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    # Define transforms for training (with augmentations) and test (no augmentations)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # CIFAR-100 mean/std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    # loading and split the training data into training and validation sets
    full_train = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(full_train))  # 80% for training
    val_size = len(full_train) - train_size  # 20% for validation
    trainset, valset = random_split(full_train, [train_size, val_size])

    # DataLoaders to handle batching and shuffling
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # Test set loader (CIFAR-100 test set)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # Instantiate model and move it to the appropriate device (MPS/GPU/CPU)
    model = SimpleCNN().to(CONFIG["device"])

    print("\nModel summary:")
    print(f"{model}\n")

    # Define loss function: cross entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Define optimizer: AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)

    # Learning rate scheduler: gradually reduce learning rate using cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Initialize Weights & Biases for tracking experiments
    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0  # For saving best model based on validation accuracy

    # Training loop: run for each epoch
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()  # Update learning rate

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Save the best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()  # End W&B session

    # final Evaluation on Clean CIFAR-100 + OOD Test Sets
    import eval_cifar100
    import eval_ood

    # Load the best performing model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # evaluate clean CIFAR-100 test accuracy
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # evaluate OOD performance
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # save predictions for leaderboard submission
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

# Entry point of the script
if __name__ == '__main__':
    main()
