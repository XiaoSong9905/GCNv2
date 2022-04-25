#
# Convert a pytorch.pth model to onnx model
# Reference https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# 

import sys
import numpy as np
from torch import nn
import torch.onnx
import onnx
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms

def main():
    # Get input output file
    if ( len(sys.argv) != 3 ):
        print("Usage: python3 convert_pytorch_to_onnx.py PYTORCH_MODEL_FILENAME ONNX_MODEL_FILENAME")
        exit(1)

    PYTORCH_MODEL_FILENAME = sys.argv[1]
    ONNX_MODEL_FILENAME = sys.argv[2]

    # Load model structure and parameter
    torch_model = torch.load( PYTORCH_MODEL_FILENAME )

    # Set model to inference mode
    torch_model.eval()

    # Input to the model
    # Actual input should be fp32 in range [0,1]
    # Intel realsense : width = 640, height = 480
    batch_size = 1
    input_width = 640
    input_height = 480
    input_channel = 1 # gray image
    torch_input = torch.randn(batch_size, input_channel, input_height, input_width, requires_grad=True)

    # Run the model and get output, this is used to verify result on onnx is correct
    torch_output = torch_model( torch_input )

    # Export the model to onnx
    torch.onnx.export(torch_model,               # model being run
                      torch_input,               # model input (or a tuple for multiple inputs)
                      ONNX_MODEL_FILENAME,              # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['detector', 'descriptor'], # the model's output names
                      )

    # Load onnx model with onnx runtime and check model
    onnx_model = onnx.load(ONNX_MODEL_FILENAME)
    onnx.checker.check_model( onnx_model )

    # Inference with onnx to validate correct model output
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_FILENAME)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_input = {ort_session.get_inputs()[0].name: to_numpy( torch_input )}
    ort_output = ort_session.run(None, ort_input)


if __name__ == '__main__':
    main()