# Compile the model for deployment using TorchScript
import torch

try:
    # Load the trained model
    model = torch.load('trained-models/transfer-learning-model.pth')
    model.eval()

    # Create an example input tensor needed by the jit tracer
    example = torch.rand(1, 3, 224, 224)

    # Generate TorchScript by tracing
    with torch.no_grad():
        scripted_model = torch.jit.trace(model, example)

    # Save the compiled model
    torch.jit.save(scripted_model, 'trained-models/scripted_model.pth')

    print("Model compiled and saved successfully.")

except Exception as e:
    print("Error occurred during model compilation:", e)
