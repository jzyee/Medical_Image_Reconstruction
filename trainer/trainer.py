import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.nn import L1Loss, MSELoss
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models.unet_beamformer import UNetBeamformer
from monai.networks.layers import HilbertTransform
from models.transformer_cnn import TransformerCNN

class Trainer():

    @staticmethod
    def norm(x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    def __init__(self, model, dataset, args, split=0.8):
        self.bs = args.bs
        self.lr = args.lr
        self.loss = args.loss
        self.model = model()
        # self.criterion = L1Loss() 
        
        if (self.loss == 'MAE'):
            self.criterion = L1Loss()
        elif (self.loss == 'MSE'):
            self.criterion = MSELoss()
        else:
            raise ValueError(f'self.loss value: {self.loss} is not of the correct type')

        # Train/Validation split
        self.train_size = int(len(dataset) * split)
        self.valid_size = len(dataset) - self.train_size
        self.train_set, self.valid_set = random_split(
            dataset, [self.train_size, self.valid_size], generator=torch.Generator().manual_seed(42)
        )

        # DataLoader with configurable number of workers
        self.train_data = DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        self.valid_data = DataLoader(self.valid_set, batch_size=self.bs, num_workers=args.num_workers, pin_memory=True)

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.save_paths = args.save

    def train(self, epochs, run_no, ground_truth_logged=False):
        # Set up directories and logging
        save_paths = self.save_paths
        os.makedirs(os.path.join(save_paths, "chkpt/", 'iter_' + str(run_no)), exist_ok=True)
        os.makedirs(os.path.join(save_paths, "logs/"), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_paths, 'logs/iter_' + str(run_no)))

        chkpt = os.path.join(save_paths, "chkpt", 'iter_' + str(run_no), "model.pt")
        best = os.path.join(save_paths, "chkpt", 'iter_' + str(run_no), "best.pt")
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scaler = GradScaler()
        #scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.1, verbose=True)


        # Load checkpoint if exists
        if os.path.exists(chkpt):
            checkpoint = torch.load(chkpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
            start = checkpoint['epoch']
            train_step = checkpoint['train_step']
            val_step = checkpoint['val_step']
            threshold = checkpoint['loss']
        else:
            start = 0
            train_step = 0
            val_step = 0
            threshold = 1000
            ground_truth_logged = True

        # Training loop
        for epoch in range(start, epochs):
            self.model.train()
            train_loss = 0
            val_loss = 0

            for i, batch in enumerate(self.train_data):
                input_data = batch['input'].to(self.device)
                output = batch['output'].to(self.device)

                with autocast():
                    # predicts the weights with the input data
                    pred = self.model(input_data)
                    beamformed = torch.mul(pred, input_data)
                    # collapse in the channel dim
                    beamformed_sum = torch.sum(beamformed, axis=1)
                    # for the amplitude
                    beamformed_sum = HilbertTransform(axis=1)(beamformed_sum)
                    envelope = torch.abs(beamformed_sum)
                    # low compression to see the smaller values and the higher values together
                    imPred = 20 * torch.log10(envelope / torch.clip(torch.max(envelope), min=1e-8))
                    # compare the predicition w the ground truth
                    loss = self.criterion(imPred, output) / 4  # Gradient accumulation over 4 steps

                # loss.backward()  # Backpropagation
                # optimizer.step()  # Update the weights

                scaler.scale(loss).backward()

                # Accumulate gradients and update every 4 steps
                if (i + 1) % 4 == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * 4  # Undo the division for logging purposes

                train_step += 1

            train_loss /= len(self.train_set)

            # Validation loop
            val_loss, ground_truth_logged = self.validate(val_step, writer, epoch, ground_truth_logged)

            print(f'Epoch = {epoch:3d}, Training Loss = {train_loss:.3f}, Validation Loss = {val_loss:.3f}')
            writer.add_scalar('Train Loss', train_loss, global_step=epoch)
            writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
            # Learning rate scheduler step
            scheduler.step(val_loss)

            # Save checkpoint
            if epoch % 5 == 0 or val_loss < threshold:
                print("Saving model checkpoint...")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': val_loss,
                    'train_step': train_step,
                    'val_step': val_step,
                }, chkpt)

            # Save the best model
            if val_loss < threshold:
                print("Saving best model weights")
                threshold = val_loss
                torch.save(self.model.state_dict(), best)

            writer.flush()

        writer.close()

    def validate(self, val_step, writer, epoch, ground_truth_logged):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.valid_data:
                input_data = batch['input'].to(self.device)
                output = batch['output'].to(self.device)

                with autocast():
                    pred = self.model(input_data)
                    beamformed = torch.mul(pred, input_data)
                    beamformed_sum = torch.sum(beamformed, axis=1)
                    beamformed_sum = HilbertTransform(axis=1)(beamformed_sum)
                    envelope = torch.abs(beamformed_sum)
                    imPred = 20 * torch.log10(envelope / torch.clip(torch.max(envelope), min=1e-8))
                    loss = self.criterion(imPred, output)

                val_loss += loss.item()
                val_step += 1

        val_loss /= len(self.valid_set)

        # Now save the best image to see the iteration

        # Convert to numpy for visualization
        imPred = self.norm(imPred)
        output = self.norm(output)
        imPred = imPred.detach().cpu().numpy()
        output = output.detach().cpu().numpy()


        # TensorBoard Image logging
        # Log ground truth only once, during the first batch
        if ground_truth_logged:
            writer.add_image('GT', output[0], global_step=epoch, dataformats='HW')
        writer.add_image('Pred'+str(epoch), imPred[0], global_step=epoch, dataformats='HW')

        # Visualization using matplotlib
        # Display the ground truth and prediction for visual reference, only once
        if ground_truth_logged:
            plt.imshow(output[0], cmap='gray', aspect='auto')
            plt.axis('off')
            plt.title('Ground Truth')
            plt.show()
            ground_truth_logged = False  # Ensure ground truth is logged once

        plt.imshow(imPred[0], cmap='gray', aspect='auto')
        plt.axis('off')

        return val_loss,ground_truth_logged


