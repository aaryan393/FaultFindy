import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    def __init__(self, num_classes=2):
        super(NeuralNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # Freeze all layers except the fully connected layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Modify the fully connected layers to match the number of output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Load the model
model = NeuralNet()
# Specify the path to the saved state dictionary
state_dict_path = 'densenet_finetuned.pth'

# Load the state dictionary
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))

# Filter out unnecessary keys from the state dictionary
# This is required if the state dictionary was saved with a different model structure
# than the one we are currently using
#filtered_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model.' in k}

# Load the filtered state dictionary into the model
#model.load_state_dict(filtered_state_dict)

# Set the model to evaluation mode
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class_names=['defective', 'good']

def predict(image_path, models):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

# Example usage
result = predict('defective-tires-cause-accidents-min.jpg',models=model)
print('Predicted class:', result)
