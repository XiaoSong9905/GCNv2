#
# Test if the given model in GCNv2_SLAM is functioning
#
import sys
import torch

def main():
    # Get input output file
    if ( len(sys.argv) != 2 ):
        print("Usage: python3 test_gcnv2_model.py TORCHSCRIPT_MODEL_FILENAME")
        exit(1)

    TORCHSCRIPT_MODEL_FILENAME = sys.argv[1]

    # Define input
    batch_size = 1
    input_width = 640
    input_height = 480
    input_channel = 1 # gray image
    torch_input = torch.randn(batch_size, input_channel, input_height, input_width, requires_grad=True, device=torch.device('cuda:0') )

    # Convert to torchscript model
    # Our model do not have fancy branching inside forward() call.
    # It's enough to just use script to convert
    print("Start loading model\n")
    torchscript_model = torch.jit.load( TORCHSCRIPT_MODEL_FILENAME ).to( torch.device('cuda:0'))

    # Run torchscript model
    print("Start running model\n")
    torchscript_output = torchscript_model( torch_input )

    print("After running model\n")

if __name__ == '__main__':
    main()