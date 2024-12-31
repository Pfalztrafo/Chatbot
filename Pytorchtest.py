import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(f"Gerät: {torch.device('cpu')}")
