[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Vision LLama
Implementation of VisionLLaMA from the paper: "VisionLLaMA: A Unified LLaMA Interface for Vision Tasks" in PyTorch and Zeta. [PAPER LINK](https://arxiv.org/abs/2403.00522)


## install
`$ pip install vision-llama`


## usage
```python

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

```



# License
MIT

## Citation
```bibtex
@misc{chu2024visionllama,
    title={VisionLLaMA: A Unified LLaMA Interface for Vision Tasks}, 
    author={Xiangxiang Chu and Jianlin Su and Bo Zhang and Chunhua Shen},
    year={2024},
    eprint={2403.00522},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## todo
- [ ] Implement the AS2DRoPE rope, might just use axial rotary embeddings instead, my implementation is really bad
- [x] Implement the GSA attention, i implemented it but's bad

