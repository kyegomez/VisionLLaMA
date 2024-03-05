import torch
from vision_llama.main import VisionLlamaPyramidBlock

# Create a random tensor of shape (1, 3, 224, 224)
x = torch.randn(1, 3, 224, 224)

# Create an instance of the VisionLlamaPyramidBlock model with the specified parameters
model = VisionLlamaPyramidBlock(768, 12, 3, 12)

# Print the shape of the output tensor when x is passed through the model
print(model(x))
