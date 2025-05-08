import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score  
import matplotlib.pyplot as plt
import time
start_time = time.time() 

device = torch.device("cuda")

# I defined transform operations to make the image data suitable for the model.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset is created and train/val separation is done
dataset = datasets.ImageFolder(root="flowers/train", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = dataset.classes
num_classes = len(class_names)

vgg16 = models.vgg16(pretrained=True)

# I froze the first 5 layers.
for param in vgg16.features[:5].parameters():
    param.requires_grad = False
    
# I edited the classifier section according to the assignment
vgg16.classifier[6] = nn.Linear(4096, num_classes)
vgg16 = vgg16.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=0.0001)


num_epochs = 10
for epoch in range(num_epochs):
    vgg16.train()
    total_train_loss = 0
    all_train_preds = []
    all_train_labels = []

    # The parameters of the model are updated by iterating over all training data.
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = accuracy_score(all_train_labels, all_train_preds)  

    # We put the model in evaluation mode and test it on valid data without calculating the gradient.
    vgg16.eval()
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)  

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}")
    torch.save(vgg16.state_dict(), "vgg16_finetune.pth")
    
end_time = time.time() 
total_time = end_time - start_time

print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

print(classification_report(all_val_preds, all_val_labels, target_names=class_names))


def visualize_vggfinetune(model, num_filters=6):
    model.eval()
    layers = [0, 10, 28]  # Blocks 1, 3, 5 in VGG16

    sample_img, _ = next(iter(train_loader))
    sample_img = sample_img[0].unsqueeze(0).to(device)

    x = sample_img.clone()
    activations = []

    # Visualize activations.
    for idx in range(max(layers) + 1):
        x = model.features[idx](x)
        if idx in layers:
            activations.append(x.clone())

   
    fig, axes = plt.subplots(len(activations), num_filters + 1, figsize=(20, 15))
    for i in range(len(activations)):
        
        axes[i, 0].imshow(sample_img.squeeze().permute(1, 2, 0).cpu() * 0.5 + 0.5)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')

        activation = activations[i].squeeze().detach().cpu()
        for j in range(num_filters):
            if j < activation.shape[0]:
                axes[i, j+1].imshow(activation[j], cmap='viridis')
                axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()

visualize_vggfinetune(vgg16, num_filters=6)