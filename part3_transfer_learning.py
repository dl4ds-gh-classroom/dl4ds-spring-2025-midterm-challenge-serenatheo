import os  #  handle file paths and directories
import numpy as np  # for numerical computations and seeding
import pandas as pd  # For working with CSVs, e.g., creating submission files
import urllib.request  # Used for downloading external resources (not directly used here)
import torch  # core PyTorch library
import torch.nn as nn  # For defining neural network layers
import torch.optim as optim  # For optimizers like AdamW
import torchvision  # Provides pretrained models and CIFAR100 dataset
import torchvision.transforms as transforms  # For data preprocessing and augmentation
from torch.utils.data import DataLoader, random_split, Subset, TensorDataset  # Dataset and loading utilities
from tqdm.auto import tqdm  # Used for displaying progress bars in loops
import wandb  # Weights & Biases for experiment tracking and visualizations

# Model Definition (pretrained ResNet50 with fine-tuned final layer)
class ResNet50Transfer(nn.Module):
    def __init__(self):
        super(ResNet50Transfer, self).__init__()  # calling the superclass constructor
        self.model = torchvision.models.resnet50(pretrained=True)  # Load pretrained ImageNet ResNet50

        # Replace the final FC layer to suit CIFAR-100 (100 classes)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),  # Apply dropout for regularization to prevent overfitting
            nn.Linear(self.model.fc.in_features, 100)  # Output layer mapping to 100 class logits
        )

    def forward(self, x):
        return self.model(x)  # Pass input through the ResNet50 model

# One Epoch Training Loop
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]  # Use device from config
    model.train()  # Set model to training mode (enables dropout and batch norm)

    running_loss = 0.0  # Accumulates loss for reporting
    correct = 0  # Total correct predictions
    total = 0  # Total number of examples seen

    # Progress bar for visualization
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        # Move inputs and labels to device (CPU/GPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward + backward + optimize
        optimizer.zero_grad()  # Reset gradients before backprop
        outputs = model(inputs)  # Forward pass through model
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation (compute gradients)
        optimizer.step()  # Update weights using gradients

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get predicted class index
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()  # Compare prediction to ground truth

        # Display current loss and accuracy on progress bar
        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    # Return average loss and overall accuracy for epoch
    return running_loss / len(trainloader), 100. * correct / total

# One Epoch Validation Loop
def validate(model, valloader, criterion, device):
    model.eval()  # Set model to eval mode (disable dropout/batchnorm updates)

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation needed
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass only
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  # Compare with true label

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    # Return average validation loss and accuracy
    return running_loss / len(valloader), 100. * correct / total

# Main Function to Train and Evaluate the Model
def main():
    # Configuration dictionary to control all major parameters
    CONFIG = {
        "model": "ResNet50",  # Model name (used in wandb)
        "batch_size": 64,  # Number of examples per batch
        "learning_rate": 0.0005,  # Initial learning rate
        "epochs": 30,  # Total number of training epochs
        "num_workers": 4,  # Number of worker threads for loading data
        "device": "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu",
        "data_dir": "./data",  # Path to CIFAR-100 data
        "ood_dir": "./data/ood-test",  # Path to out-of-distribution data
        "wandb_project": "sp25-ds542-challenge",  # Name of wandb project
        "seed": 42  # Seed for reproducibility
    }

    # Set all random seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    import random
    random.seed(CONFIG["seed"])
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed(CONFIG["seed"])

    # Print configuration dictionary for verification
    print("\nCONFIG Dictionary:")
    import pprint
    pprint.pprint(CONFIG)

    # --------------------------
    # Data Preprocessing
    # --------------------------
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # dataset Loading and Splitting

    # download entire CIFAR-100 training set 
    full_train = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False, download=True, transform=transform_test)

    # Split into training and validation sets using an 80/20 ratio
    train_size = int(0.8 * len(full_train))  # 80% for training
    val_size = len(full_train) - train_size  # Remaining 20% for validation
    train_subset, val_subset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG["seed"])  # Ensure consistent splitting
    )

    # Reload the dataset with appropriate transforms
    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True, transform=transform_test)

    # Apply the split indices to the transformed datasets
    trainset = Subset(trainset, train_subset.indices)
    valset = Subset(valset, val_subset.indices)

    # DataLoaders
    # Dataloader for training: uses shuffle to help generalization
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])

    # Validation: no need to shuffle since we don’t train on it
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # Test set: used for final evaluation
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # --------------------------
    # Model, Loss, Optimizer, Scheduler
    # --------------------------
    model = ResNet50Transfer().to(CONFIG["device"])  # Move model to device (GPU or CPU)

    print("\nModel summary:\n", model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # WandB Logging Setup
    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["model"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    # Final Evaluation + Submission File
    import eval_cifar100
    import eval_ood

    model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))
    model.eval()

    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv("submission_ood_transfer_resnet50.csv", index=False)
    print("submission_ood_transfer_resnet50.csv created successfully.")

# entry point for data pipeline
if __name__ == '__main__':
    main()
