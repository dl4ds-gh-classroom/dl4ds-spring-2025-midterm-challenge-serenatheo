import os  # Handles file path operations
import numpy as np  # For numerical computations and seeding
import pandas as pd  # For working with CSVs, e.g., to save submission files
import urllib.request  # To download external resources (not directly used)
import torch  # PyTorch core functionality
import torch.nn as nn  # Defines neural network layers
import torch.optim as optim  # Optimizers like AdamW
import torchvision  # PyTorch's vision library (models, datasets)
import torchvision.transforms as transforms  # Image transformation utilities
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset  # Data loading and splitting utilities
from tqdm.auto import tqdm  # Progress bar for training/validation
import wandb  # Weights & Biases for experiment tracking

# Define Part 3 model class for transfer learning using pretrained ResNet50
class ResNet50Transfer(nn.Module):  # Inherit from PyTorch's Module class
    def __init__(self):  # Constructor for initializing layers
        super(ResNet50Transfer, self).__init__()  # Call parent constructor
        self.model = torchvision.models.resnet50(pretrained=True)  # Load pretrained ResNet50 from ImageNet
        self.model.fc = nn.Sequential(  # Replace original fully connected (fc) layer
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(self.model.fc.in_features, 100)  # Output layer for 100 CIFAR-100 classes
        )  # End of layer replacement

    def forward(self, x):  # Define forward pass
        return self.model(x)  # Pass input through modified ResNet50

# One epoch of training
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):  # Train function
    device = CONFIG["device"]  # Get device (CPU/GPU)
    model.train()  # Set model to training mode

    running_loss = 0.0  # Track cumulative loss
    correct = 0  # Track correct predictions
    total = 0  # Track total samples

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)  # Progress bar

    for i, (inputs, labels) in enumerate(progress_bar):  # Loop through batches
        inputs, labels = inputs.to(device), labels.to(device)  # Send data to device
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss
        _, predicted = outputs.max(1)  # Get predicted labels
        total += labels.size(0)  # Update total
        correct += predicted.eq(labels).sum().item()  # Update correct

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})  # Show stats

    return running_loss / len(trainloader), 100. * correct / total  # Return average loss and accuracy

# One epoch of validation
def validate(model, valloader, criterion, device):  # Validate function
    model.eval()  # Set to evaluation mode

    running_loss = 0.0  # Accumulate validation loss
    correct = 0  # Correct prediction counter
    total = 0  # Total counter

    with torch.no_grad():  # Disable gradients
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)  # Progress bar
        for i, (inputs, labels) in enumerate(progress_bar):  # Loop through batches
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss

            running_loss += loss.item()  # Accumulate
            _, predicted = outputs.max(1)  # Get predicted class
            total += labels.size(0)  # Total samples
            correct += predicted.eq(labels).sum().item()  # Correct predictions

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})  # Show stats

    return running_loss / len(valloader), 100. * correct / total  # Return validation metrics

# Main training pipeline
def main():  # Entry function
    CONFIG = {  # Define config
        "model": "ResNet50",
        "batch_size": 64,
        "learning_rate": 0.0005,
        "epochs": 30,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42
    }

    torch.manual_seed(CONFIG["seed"])  # PyTorch seed
    np.random.seed(CONFIG["seed"])  # NumPy seed
    import random  # Python seed
    random.seed(CONFIG["seed"])  # Set it
    if CONFIG["device"] == "cuda": torch.cuda.manual_seed(CONFIG["seed"])  # CUDA seed

    print("\nCONFIG Dictionary:")  # Log
    import pprint  # Pretty printer
    pprint.pprint(CONFIG)  # Show config

    IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Normalize mean
    IMAGENET_STD = (0.229, 0.224, 0.225)  # Normalize std

    transform_train = transforms.Compose([  # Training transforms
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_test = transforms.Compose([  # Validation/test transforms
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True)  # Full training set
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)  # Test set

    train_size = int(0.8 * len(full_train))  # 80% for training
    val_size = len(full_train) - train_size  # 20% for validation
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(CONFIG["seed"]))  # Split set

    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, transform=transform_train)  # With transforms
    valset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, transform=transform_test)

    trainset = Subset(trainset, train_subset.indices)  # Subset for training
    valset = Subset(valset, val_subset.indices)  # Subset for validation

    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])  # Train loader
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])  # Val loader
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])  # Test loader

    model = ResNet50Transfer().to(CONFIG["device"])  # Instantiate model and move to device
    print("\nModel summary:\n", model)  # Print architecture

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Loss function with label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)  # Optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])  # LR scheduler

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)  # Init wandb
    wandb.watch(model)  # watching model gradients

    best_val_acc = 0.0  # tracking best val acc
    for epoch in range(CONFIG["epochs"]):  # Epoch loop
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)  # Train loss and accuracy metrics
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])  # Validation loss and accuracy metrics
        scheduler.step()  # Step scheduler

        wandb.log({  # Log metrics
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_acc > best_val_acc:  # If improved
            best_val_acc = val_acc  # Update
            torch.save(model.state_dict(), "best_model.pth")  # Save
            wandb.save("best_model.pth")  # Save to wandb

    wandb.finish()  # End wandb session

    import eval_cifar100  # CIFAR evaluation script
    import eval_ood  # OOD eval script

    model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))  # Load best model
    model.eval()  # Eval mode

    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])  # Run clean eval
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")  # Print result

    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)  # OOD prediction
    submission_df = eval_ood.create_ood_df(all_predictions)  # Create dataframe
    submission_df.to_csv("submission_ood_transfer_resnet50.csv", index=False)  # Save to CSV
    print("submission_ood_transfer_resnet50.csv created successfully.")  # Confirm success

if __name__ == '__main__':  # Script entry point
    main()  # Run main
