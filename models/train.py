import torch

from arch import GCNv2, GCNv2_tiny

def train_GCNv2():
    # Instanize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("model train on {}".format(device))
    model = GCNv2().to(device)    

    # Add train code here


    # Save model (parameter and structure)
    # NOTE: do not use model.state_dict() here since we want to store the structure
    torch.save(model, "GCNv2.pth")

    # Since we store the weight and sturctue of model, we can load all model information
    # model = torch.load("GCNv2.pth")

if __name__ == '__main__':
    train_GCNv2()