import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import random_split

import time
start_time = time.time()

# I defined transform operations to make the image data suitable for the model.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

device = torch.device("cuda")

# Dataset is created and train/val separation is done
dataset = datasets.ImageFolder(root="flowers/train", transform=transform)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.10)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.20)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.30)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(0.40)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout(0.50)

        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 5)  # 5 classes (flower types)

    def forward(self, x):
        # Conv1 -> ReLU -> Pooling -> Dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)

        x = self.pool(F.relu(self.conv5(x)))
        x = self.dropout5(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x

model = CustomCNN().to(device)

#I used CrossEntropyLoss because it is multi-class classification.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 4
best_val_loss = float('inf')
epochsEarly= 0

num_epochs = 35
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    all_train_preds = [] # I created a list to store all the predictions.
    all_train_labels = [] # I created a list to store all the labels.

    # Training the model by iterating through mini-batches, performing forward and backward passes, 
    # calculating the loss, and updating the model's weights for images, labels in train_loader : 
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(images) 
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step() 

        train_loss += loss.item()  
        _, preds = torch.max(outputs, 1) 
        all_train_preds.extend(preds.cpu().numpy())  
        all_train_labels.extend(labels.cpu().numpy())  

    train_accuracy = accuracy_score(all_train_labels, all_train_preds) * 100

    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(all_val_labels, all_val_preds) * 100

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochsEarly = 0  
        torch.save(model.state_dict(), "best_model_cnn.pth")
    else:
        epochsEarly += 1

    if epochsEarly >= patience:
        print(f"Early stopping at epoch {epoch+1}.")
        break

# Classification Report
print(classification_report(all_val_preds, all_val_labels, target_names=dataset.classes))
end_time = time.time()
total_time = end_time - start_time

print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Feature visualization for first, third, and fifth layers
def visualize_cnn(model, train_loader, device, num_filters=6):
    model.eval()
    layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
    selected_indices = [0, 2, 4]  

    sample_img, _ = next(iter(train_loader))
    sample_img = sample_img[0].unsqueeze(0).to(device)

    x = sample_img.clone()
    activations = []
    
    # Visualize activations.
    for idx, layer in enumerate(layers):
        x = layer(x)
        if idx in selected_indices:
            activations.append(x.clone())

    fig, axes = plt.subplots(len(activations), num_filters + 1, figsize=(20, 5 * len(activations)))

    for i, activation in enumerate(activations):
        activation = activation.squeeze().detach().cpu()
        img = sample_img.squeeze().permute(1, 2, 0).cpu() * 0.5 + 0.5  # normalize

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input Image - Layer {selected_indices[i]+1}")
        axes[i, 0].axis("off")

        for j in range(num_filters):
            if j < activation.shape[0]:
                fmap = activation[j]
                fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)  # normalize
                axes[i, j + 1].imshow(fmap, cmap='viridis')
                axes[i, j + 1].axis("off")

    plt.tight_layout()
    plt.show()
visualize_cnn(model, train_loader, device, num_filters=6)