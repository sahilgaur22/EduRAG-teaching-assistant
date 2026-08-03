import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# import whisper
# model = whisper.load_model("large-v2", device="cuda")
# model = whisper.load_model("turbo", device="cuda")
# print(next(model.parameters()).device)

# import whisper
# print(whisper.__file__)
# print(whisper.__version__)