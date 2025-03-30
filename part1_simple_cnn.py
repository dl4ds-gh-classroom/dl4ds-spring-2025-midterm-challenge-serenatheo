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


# Define a simple CNN model for classifying CIFAR-100 images
class SimpleCNN(nn.Module):  # Define a neural network class
    def __init__(self):  # Initialize layers
        super(SimpleCNN, self).__init__()  # Call superclass constructor

        # Feature extractor block using convolutional layers
        self.features = nn.Sequential(  # Stack layers sequentially
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # First convolutional layer
            nn.ReLU(),  # Apply ReLU activation
            nn.MaxPool2d(2, 2),  # Downsample spatial dimensions by 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Second convolution
            nn.ReLU(),  # Apply ReLU activation
            nn.MaxPool2d(2, 2),  # Downsample again

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Third convolution
            nn.ReLU(),  # Apply ReLU activation
            nn.MaxPool2d(2, 2),  # Downsample to final feature map size

            nn.Flatten(),  # Flatten output to vector
            nn.Dropout(0.3)  # Apply dropout for regularization
        )

        # Classifier block with fully connected layers
        self.classifier = nn.Sequential(  # Stack linear layers for classification
            nn.Linear(128 * 4 * 4, 256),  # First dense layer
            nn.ReLU(),  # Activation function
            nn.Linear(256, 100)  # Output layer for 100 classes
        )

    def forward(self, x):  # Define forward pass
        x = self.features(x)  # Extract features
        x = self.classifier(x)  # Pass features to classifier
        return x  # Return output logits

# Training function for one epoch
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]  # Get device (CPU, GPU, or MPS)
    model.train()  # Set model to training mode

    running_loss = 0.0  # Track total loss
    correct = 0  # Track correct predictions
    total = 0  # Track number of samples

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)  # showing training progress

    for i, (inputs, labels) in enumerate(progress_bar):  # Iterate through batches
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()  # Accumulate loss
        _, predicted = outputs.max(1)  # Get predicted class
        total += labels.size(0)  # Increment total samples
        correct += predicted.eq(labels).sum().item()  # Increment correct predictions

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})  # updating progress bar

    train_loss = running_loss / len(trainloader)  # Average training loss
    train_acc = 100. * correct / total  # Average training accuracy
    return train_loss, train_acc  # Return training metrics

# Validation function for evaluation on validation set
def validate(model, valloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0  # Track total validation loss
    correct = 0  # Track correct predictions
    total = 0  # Track number of samples

    with torch.no_grad():  # Disable gradient computation
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)  # showing validation progress
        for i, (inputs, labels) in enumerate(progress_bar):  # Iterate through validation batches
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item()  # Accumulate loss
            _, predicted = outputs.max(1)  # Get predicted class
            total += labels.size(0)  # Increment total samples
            correct += predicted.eq(labels).sum().item()  # Increment correct predictions

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})  # Update progress

    val_loss = running_loss / len(valloader)  # Average validation loss
    val_acc = 100. * correct / total  # Average validation accuracy
    return val_loss, val_acc  # Return validation metrics

# Main training and evaluation function
def main():
    CONFIG = {  # Define training configuration dictionary
        "model": "SimpleCNN",  # Model name
        "batch_size": 64,  # Mini-batch size
        "learning_rate": 0.005,  # Initial learning rate
        "epochs": 50,  # Number of training epochs
        "num_workers": 4,  # Number of CPU threads for data loading
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",  # Pick best device
        "data_dir": "./data",  # Directory to save data
        "ood_dir": "./data/ood-test",  # Directory for out-of-distribution test data
        "wandb_project": "sp25-ds542-challenge",  # W&B project name
        "seed": 42  # Random seed for reproducibility
    }

    import pprint  # For pretty printing config
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)  # Print the configuration

    # Define transforms for data augmentation during training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random flip for augmentation
        transforms.RandomCrop(32, padding=4),  # Random crop
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Add color noise
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # Normalize to CIFAR-100 stats
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))  # Normalize
    ])

    # Load CIFAR-100 training dataset
    full_train = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True, transform=transform_train)
    train_size = int(0.8 * len(full_train))  # 80% training split
    val_size = len(full_train) - train_size  # 20% validation split
    trainset, valset = random_split(full_train, [train_size, val_size])  # splitting dataset

    # creating DataLoaders for training and validation
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # loading test set
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = SimpleCNN().to(CONFIG["device"])  # instantiate and move model to device
    print("\nModel summary:")
    print(f"{model}\n")  # printing model structure

    criterion = nn.CrossEntropyLoss()  # Define loss function
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)  # Use AdamW optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])  # Learning rate scheduler

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)  # Start W&B session
    wandb.watch(model)  # Monitor gradients and parameters

    best_val_acc = 0.0  # Track best validation accuracy

    for epoch in range(CONFIG["epochs"]):  # Run training loop
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)  # Train model
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])  # Validate model
        scheduler.step()  # Update learning rate

        wandb.log({  # Log metrics to W&B
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_acc > best_val_acc:  # Save best model checkpoint
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()  # Finish logging session

    import eval_cifar100  # Import evaluation script for clean test data
    import eval_ood  # Import evaluation script for OOD test

    model.load_state_dict(torch.load("best_model.pth"))  # Load best model
    model.eval()  # Set model to evaluation mode

    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])  # evaluating CIFAR-100
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")  # Print accuracy
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)  # evaluating on OOD test set

    submission_df_ood = eval_ood.create_ood_df(all_predictions)  # creating submission file
    submission_df_ood.to_csv("submission_ood.csv", index=False)  # saving as CSV
    print("submission_ood.csv created successfully.")  # confirmation

# running script if executed directly
if __name__ == '__main__':
    main()  # calling main function
