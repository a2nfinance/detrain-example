import torch
import torch.nn as nn
import time
import os
from detrain.ppl.args_util import get_args
from detrain.tp.train_eval import train_eval
from detrain.tp.model_utils import get_tp_model, save_model
from detrain.ppl.dataset_util import get_torchvision_dataset
import torch.optim as optim
from base_model import NeuralNetwork
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor import Shard

if __name__=="__main__":
    args = get_args()
    world_size = int(os.environ["WORLD_SIZE"])
    # Get args
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    device = "cpu"

    # Check devices
    if (args.gpu is not None):
        device = "cuda"
    
    # Define optimizer & loss_fn
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer_class = optim.SGD
    model = NeuralNetwork().to(device)

   
    mesh_shape = (world_size, )
    tp_model = get_tp_model(model, {
        "in_proj": ColwiseParallel(
            use_local_output=False,
        ),
        "linear1": RowwiseParallel(
            use_local_output=False,
        ),
        "out_proj": ColwiseParallel(

            use_local_output=False,
        ),
    } , device, mesh_shape)

    
    # Create an optimizer for the parallelized module.
    optimizer = torch.optim.SGD(tp_model.parameters(), lr=lr)
    
    # Dataloaders

    (train_dataloader, test_dataloader) = get_torchvision_dataset("MNIST", batch_size)

    tik = time.time()
    train_eval(
        tp_model, 
        train_dataloader, 
        test_dataloader, 
        loss_fn, 
        optimizer, 
        epochs, 
        batch_size,
        device
    )
    tok = time.time()
    print(f"Execution time = {tok - tik}")
    save_model(model, args.model_name)