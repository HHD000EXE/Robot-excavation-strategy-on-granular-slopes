import torch
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_zero, weight_non_zero):
        super(WeightedMSELoss, self).__init__()
        self.weight_zero = weight_zero
        self.weight_non_zero = weight_non_zero

    def forward(self, input, target):
        mask_tensor = torch.where(abs(target) < 0.2, torch.tensor(weight_zero), torch.tensor(weight_non_zero))
        criterion = nn.MSELoss(reduction='none')  # Use 'none' to avoid reducing the loss yet
        mse_loss = criterion(input, target)
        weighted_loss = torch.mean(mask_tensor*mse_loss)

        return weighted_loss

# Example usage:
weight_zero = 1  # Weight for samples with label [0, y] in the first row
weight_non_zero = 5  # Weight for samples with non-zero label in the first row

# Initialize the weighted loss function
criterion = WeightedMSELoss(weight_zero, weight_non_zero)


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, train=True, test_split=0.2):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.train = train
        self.test_split = test_split

        self.image_files = os.listdir(image_folder)

        if self.train:
            # Split the data into training and test sets
            train_files, test_files = train_test_split(self.image_files, test_size=self.test_split, random_state=42)
            self.image_files = train_files
        else:
            # Use the remaining files for testing
            train_files, test_files = train_test_split(self.image_files, test_size=self.test_split, random_state=42)
            self.image_files = test_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name)

        # Get the corresponding label CSV file for this image
        label_file_name = os.path.splitext(self.image_files[idx])[0] + '.csv'
        label_file_path = os.path.join(self.label_folder, label_file_name)

        label_data = pd.read_csv(label_file_path)  # Load the label data from the associated CSV file
        label_data = label_data[['fx', 'fy']]
        label = label_data.to_numpy(dtype='float32')  # Assuming labels are in the CSV file

        if self.transform:
            image = self.transform(image)
        # Convert NumPy array to a PyTorch tensor
        label = torch.from_numpy(label).to(device)
        label = label.t()
        # Reshape the label tensor to match the shape of the output tensor
        reshaped_label = label.reshape(2, 50, 50)
        return image.to(device), reshaped_label.to(device)


class CNNModel(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        # CNN layers for image processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Final convolutional layers for generating two-channel images
        self.output = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, image, scalar):
        # Processing image through CNN layers
        x = self.conv1(image)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)

        # Expand scalar to match image dimensions
        scalar_expanded = scalar.unsqueeze(-1).unsqueeze(-1).expand_as(image)

        # Concatenate processed image with the scalar input
        concatenated_input = torch.cat((x, scalar_expanded), dim=1)

        # Generating two-channel output images
        output = self.output(concatenated_input)

        return output


#Define data transformations
transform = transforms.Compose([transforms.Resize((1000, 1000)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

# Paths to your image folder and label folder
image_folder = 'train_images'
label_folder = 'PIV_vectors'

# Create custom dataset
train_dataset = CustomDataset(image_folder, label_folder, transform, train=True)
test_dataset = CustomDataset(image_folder, label_folder, transform, train=False)

# Create a data loader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Initialize the model
model = CNNModel().to(device)

# Define the loss function and optimizer
# criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Save the trained model if needed
torch.save(model.state_dict(), 'trained_model.pth')
model.load_state_dict(torch.load('trained_model.pth'))  # Load the trained model

# Set the model to evaluation mode
model.eval()

# Define a criterion for evaluation (e.g., Mean Squared Error)
# criterion = nn.MSELoss()

# Testing loop
total_loss = 0.0
total_samples = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         ## visualize the difference between the predcitions and labels
#         output_array = outputs.detach().cpu().numpy()  # Detach from gradients and move to CPU
#         label_array = labels.detach().cpu().numpy()
#         input_image_array = images.detach().cpu().numpy()
#         sample_index = 0
#         # Get the predicted and label data for the selected sample
#         prediction = output_array[sample_index]
#         label = label_array[sample_index]
#         sample_image = input_image_array[sample_index][0]
#         plt.figure(figsize=(20, 5))
#         # Visualize the differences for both channels
#         for channel_to_visualize in range(2):  # Iterate through both channels
#             difference = abs(prediction[channel_to_visualize] - label[channel_to_visualize])
#             # Calculate combined min and max values for the current channel
#             combined_min = min(prediction[channel_to_visualize].min(), label[channel_to_visualize].min())
#             combined_max = max(prediction[channel_to_visualize].max(), label[channel_to_visualize].max())
#             # Plot the raw image
#             plt.subplot(2, 4, channel_to_visualize*4 + 1)
#             plt.title(f'input image - Channel {channel_to_visualize}')
#             plt.imshow(sample_image, cmap='gray')
#             # Plot prediction for the current channel
#             plt.subplot(2, 4, channel_to_visualize*4 + 2)
#             plt.title(f'Prediction - Channel {channel_to_visualize}')
#             plt.imshow(prediction[channel_to_visualize], cmap='gray', vmin=combined_min, vmax=combined_max)
#             plt.colorbar()
#             # Plot label for the current channel
#             plt.subplot(2, 4, channel_to_visualize*4 + 3)
#             plt.title(f'Label - Channel {channel_to_visualize}')
#             plt.imshow(label[channel_to_visualize], cmap='gray', vmin=combined_min, vmax=combined_max)
#             plt.colorbar()
#             # Plot absolute difference for the current channel
#             plt.subplot(2, 4, channel_to_visualize*4 + 4)
#             plt.title(f'Absolute Difference - Channel {channel_to_visualize}')
#             plt.imshow(difference, cmap='viridis')  # Change colormap if needed
#             plt.colorbar()
#         plt.tight_layout()
#         plt.show()
#         ## visualize the difference between the predcitions and labels
#
#         total_loss += loss.item() * labels.size(0)
#         total_samples += labels.size(0)
#
# average_loss = total_loss / total_samples
# print(f"Test Loss: {average_loss}")