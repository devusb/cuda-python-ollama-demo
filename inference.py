import torch
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model and run inference
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()

# Generate a random input tensor (drop in a real image later)
input_tensor = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(input_tensor)

# Get top predictions
weights = models.ResNet50_Weights.DEFAULT
categories = weights.meta["categories"]
top5 = torch.topk(output[0], 5)

print(f"Device: {device}")
print("Top predictions:")
for prob, idx in zip(top5.values.softmax(0), top5.indices):
    print(f"  {categories[idx]}: {prob:.1%}")
