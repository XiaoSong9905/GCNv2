import sys
import torch

from arch import GCNv2, GCNv2_tiny

def train_GCNv2():
    # Instanize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("model train on {}".format(device))
    model = GCNv2_tiny().to(device)    

    # Add train code here


    # Save model (parameter and structure)
    # NOTE: do not use model.state_dict() here since we want to store the structure
    PYTORCH_MODEL_FILENAME = sys.argv[1]
    torch.save(model, PYTORCH_MODEL_FILENAME)

    # Since we store the weight and sturctue of model, we can load all model information
    # model = torch.load(PYTORCH_MODEL_FILENAME)

if __name__ == '__main__':
    if ( len(sys.argv) != 2 ):
        print("Usage: python3 train.py PYTORCH_MODEL_FILENAME")
        exit(1)

    train_GCNv2()