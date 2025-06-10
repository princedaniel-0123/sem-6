def main():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, models
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import warnings
    import random
    import argparse
    from tqdm import tqdm
    import logging
    logging.basicConfig(filename='training.log', level=logging.INFO)

    warnings.filterwarnings('ignore')

    # Configurable parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs_initial = args.epochs
    epochs_fine_tune = 10
    learning_rate = args.lr

    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    train_image_dir = r'D:\summa\Sign-Language-Detectiom-1\train'
    valid_image_dir = r'D:\summa\Sign-Language-Detectiom-1\valid'
    test_image_dir = r'D:\summa\Sign-Language-Detectiom-1\test'
    image_size = 128
    num_classes = 99

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load CSVs
    train_df = pd.read_csv(os.path.join(train_image_dir, '_classes.csv'))
    valid_df = pd.read_csv(os.path.join(valid_image_dir, '_classes.csv'))
    test_df = pd.read_csv(os.path.join(test_image_dir, '_classes.csv'))

    # Remove duplicate columns
    train_df = train_df.loc[:, ~train_df.columns.duplicated()]
    valid_df = valid_df.loc[:, ~valid_df.columns.duplicated()]
    test_df = test_df.loc[:, ~test_df.columns.duplicated()]

    # Filter missing images
    def filter_df(df, image_dir):
        return df[df['filename'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]

    train_df = filter_df(train_df, train_image_dir)
    valid_df = filter_df(valid_df, valid_image_dir)
    test_df = filter_df(test_df, test_image_dir)

    class_names = train_df.columns[1:].tolist()
    assert len(class_names) == num_classes

    def get_class_indices(df):
        labels = df.iloc[:, 1:].values
        assert np.all(np.sum(labels, axis=1) == 1)
        return np.argmax(labels, axis=1)

    train_labels = get_class_indices(train_df)
    valid_labels = get_class_indices(valid_df)
    test_labels = get_class_indices(test_df)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Dataset
    class SignLanguageDataset(Dataset):
        def __init__(self, df, image_dir, transform=None):
            self.df = df
            self.image_dir = image_dir
            self.transform = transform
            self.labels = get_class_indices(df)
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.df.iloc[idx, 0])
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    # Transforms
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.2),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Dataset instantiation (add this before DataLoader section)
    train_dataset = SignLanguageDataset(train_df, train_image_dir, transform=train_transform)
    valid_dataset = SignLanguageDataset(valid_df, valid_image_dir, transform=val_transform)
    test_dataset = SignLanguageDataset(test_df, test_image_dir, transform=val_transform)

    # WeightedRandomSampler for class imbalance
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoaders
    num_workers = 4  # Or use multiprocessing.cpu_count() for dynamic setting

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2, min_lr=1e-6)

    # Early Stopping
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    # Training/Validation Loop
    def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, fine_tune=False):
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        early_stopper = EarlyStopping(patience=5)
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0, 0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            train_loss = running_loss / total
            train_acc = correct / total

            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            scheduler.step(val_loss)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'class_names': class_names,
                    'transform': val_transform
                }, 'model_full.pth')
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')
        return history

    # Initial training (freeze base)
    for param in model.features.parameters():
        param.requires_grad = False
    history = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs_initial)

    # Fine-tuning (unfreeze some layers)
    for param in model.features[-5:].parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    history_fine = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs_fine_tune, fine_tune=True)

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    labels = list(range(num_classes))
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, 
        labels=labels, 
        target_names=class_names, 
        zero_division=0
    ))
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'] + history_fine['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'] + history_fine['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'] + history_fine['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'] + history_fine['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    # Save model and class names
    torch.save(model.state_dict(), 'sign_language_model.pth')
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Training complete. Model saved as 'sign_language_model.pth'. Class names saved as 'class_names.txt'. Plots saved as 'confusion_matrix.png' and 'training_history.png'.")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

if __name__ == "__main__":
    main()