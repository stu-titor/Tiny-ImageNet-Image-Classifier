from flask import Flask, request, jsonify
import requests
import io
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as transforms
from timm.data import ImageNetInfo
from CNN import ImageNeuralNetwork

app = Flask(__name__)
base_dir = Path(__file__).resolve().parent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

net = ImageNeuralNetwork(64, 4, 6, 200).to(device)
net.load_state_dict(torch.load(base_dir / 'trained_net_70.12.pth', map_location = device, weights_only = True))
net.eval()

new_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2770, 0.2691, 0.2821]
    )
])

with open(base_dir / 'data/tiny-imagenet-200/wnids.txt') as file:
    class_names = sorted(line.strip() for line in file if line.strip())

image_info = ImageNetInfo(subset='imagenet-1k')

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image

def predict(image):
    tta_images = torch.cat([image, torch.flip(image, dims=[3])]).to(device)
    with torch.no_grad():
        output = net(tta_images).mean(dim=0, keepdim=True)
        _, predicted = torch.max(output, 1)
    return predicted.item()

@app.route('/classify/file', methods=['POST'])
def fileClassify():
    file = request.files['image']
    predicted = predict(load_image(file))
    wnid = class_names[predicted]
    return jsonify({"prediction": image_info.label_name_to_description(wnid)})

@app.route('/classify/url', methods=['POST'])
def urlClassify():
    url = request.get_json()['url']
    file = io.BytesIO(requests.get(url).content)
    predicted = predict(load_image(file))
    wnid = class_names[predicted]
    return jsonify({"prediction": image_info.label_name_to_description(wnid)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
