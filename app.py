from flask import Flask,render_template, request

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

class_names=['defective', 'good']

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

model = NeuralNet()
state_dict_path = 'densenet_finetuned.pth'

# Load the state dictionary
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(image_path, models):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict(img_path,models=model)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)

