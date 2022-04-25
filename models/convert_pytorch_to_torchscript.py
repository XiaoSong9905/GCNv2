#
# Convert pytorch model to torchscript
# Reference https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html 
#
import sys
import torch

def main():
    # Get input output file
    if ( len(sys.argv) != 3 ):
        print("Usage: python3 convert_pytorch_to_torchscript.py PYTORCH_MODEL_FILENAME TORCHSCRIPT_MODEL_FILENAME")
        exit(1)

    PYTORCH_MODEL_FILENAME = sys.argv[1]
    TORCHSCRIPT_MODEL_FILENAME = sys.argv[2]

    # Load model structure and parameter
    torch_model = torch.load(PYTORCH_MODEL_FILENAME)

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
    torchscript_model.save(TORCHSCRIPT_MODEL_FILENAME)

    # Load torchscript to validate
    torchscript_model_load = torch.jit.load(TORCHSCRIPT_MODEL_FILENAME)
    torchscript_output = torchscript_model_load( torch_input )

if __name__ == '__main__':
    main()