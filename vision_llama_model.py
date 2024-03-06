import torch
from vision_llama.main import VisionLlama

# Forward Tensor
x = torch.randn(1, 3, 224, 224)

# Create an instance of the VisionLlamaBlock model with the specified parameters
model = VisionLlama(
    dim=768, depth=12, channels=3, heads=12, num_classes=1000
)


# Print the shape of the output tensor when x is passed through the model
print(model(x))
