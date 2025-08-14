import sys
import os
import torch
import torchvision.datasets
import torch.optim as optim
import torchvision.transforms
import torch.utils.data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_structure import CNN

dataset_path = 'dataset/asl_dataset/'
num_epochs = 50 # Define the number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_classes = len(dataset.classes)
num_images = len(dataset)

print(f"Number of classes: {num_classes}")
print(f"Number of images: {num_images}")

# Define model
model = CNN(num_classes)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


model.to(device)

for epoch in range(num_epochs):
    print("="*30+f"Epoch {epoch+1}".center(11)+'='*30)
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
    print(f'Epoch {epoch+1}| Epoch Loss: {running_loss/(len(dataloader)*batch_size):.5f}')

print('Finished Training')
