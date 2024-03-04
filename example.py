import torch
from vision_llama import VisionLlamaBlock

# Create a random tensor of shape (1, 3, 224, 224)
x = torch.randn(1, 3, 224, 224)

# Create an instance of the VisionLlamaBlock model with the specified parameters
model = VisionLlamaBlock(768, 12, 3, 12)

# Print the shape of the output tensor when x is passed through the model
print(model(x).shape)

# Print the output tensor when x is passed through the model
print(model(x))
