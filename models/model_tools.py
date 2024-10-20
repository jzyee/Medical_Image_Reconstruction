import os
import torch
import torch.nn as nn
from monai.networks.layers import HilbertTransform
from torch.cuda.amp import autocast
from models.unet_beamformer import UNetBeamformer
from models.transformer_cnn import TransformerCNN



# Function to get the full path of the best checkpoint
def get_best_checkpoint_path(folder):
    return os.path.join(folder, 'chkpt', 'iter_1', 'best.pt')


def load_best_model(model_class, checkpoint_dir):
    best_model = model_class()
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best.pt')
    if os.path.exists(best_checkpoint_path):
        best_model.load_state_dict(torch.load(best_checkpoint_path))
        print(f"Loaded best model from {best_checkpoint_path}")
    else:
        print(f"No best model found in {checkpoint_dir}")
    return best_model


def evaluate_models(model, test_loader, test_set):
    '''
    returns the MSE and MAE loss of the model on the test set
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.eval()

    test_loss_mse = 0
    test_loss_mae = 0

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    with torch.no_grad():
        for batch in test_loader:
            input_data = batch['input'].to(device)
            ground_truth = batch['output'].to(device)
            
            # pred_mse = mse_model(input_data)
            # pred_mae = mae_model(input_data)
            
            with autocast():
                pred = model(input_data)
                beamformed = torch.mul(pred, input_data)
                beamformed_sum = torch.sum(beamformed, axis=1)
                beamformed_sum = HilbertTransform(axis=1)(beamformed_sum)
                envelope = torch.abs(beamformed_sum)
                imPred = 20 * torch.log10(envelope / torch.clip(torch.max(envelope), min=1e-8))
                mse_loss = criterion_mse(imPred, ground_truth)
                mae_loss = criterion_mae(imPred, ground_truth)

            test_loss_mse += mse_loss.item()
            test_loss_mae += mae_loss.item()

    test_loss_mse /= len(test_set)
    test_loss_mae /= len(test_set)

    return test_loss_mse, test_loss_mae