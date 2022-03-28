#
# Convert pytorch model to torchscript
# Reference https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html 
#
import torch

def main():
    # Load model structure and parameter
    torch_model = torch.load("GCNv2.pth")

    # Set model to inference mode
    torch_model.eval()

    # Define input
    batch_size = 1
    input_width = 640
    input_height = 480
    input_channel = 1 # gray image
    torch_input = torch.randn(batch_size, input_channel, input_height, input_width, requires_grad=True)

    # Run the model and get output, use to verify load model work
    torch_output = torch_model( torch_input )

    # Convert to torchscript model
    # Our model do not have fancy branching inside forward() call.
    # It's enough to just use script to convert
    torchscript_model = torch.jit.trace( torch_model, torch_input )

    # Run torchscript model
    torchscript_output = torchscript_model( torch_input )

    # Save torchscript model
    torchscript_model.save("GCNv2.pt")

    # Load torchscript to validate
    torchscript_model_load = torch.jit.load("GCNv2.pt")
    torchscript_output = torchscript_model_load( torch_input )

if __name__ == '__main__':
    main()