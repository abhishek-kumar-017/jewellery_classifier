import torch
from torchvision import transforms, models
from PIL import Image
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(config['classes']))
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return config['classes'][predicted.item()]
