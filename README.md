[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vision LLama
Implementation of VisionLLaMA from the paper: "VisionLLaMA: A Unified LLaMA Interface for Vision Tasks" in PyTorch and Zeta.


## install
`$ pip install vision-llama`


## usage
```python

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

```



# License
MIT
