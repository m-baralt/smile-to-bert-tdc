######################
### import modules ###
######################
import sys
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tqdm
import time
sys.path.append(os.getcwd())
from model2encoder.Model.m2encoder import Model_2enc
from model2encoder.Model.training_funcs import SMILESDataset2enc

#from tdc.benchmark_group import admet_group
#from tdc.single_pred import ADME

def train_model(model, train_loader, valid_loader, criterion, optimizer, 
                model_dir, num_epochs=50, patience=5, device='cuda'):
    """
    This function trains the model with early stopping and evaluates it with the validation split.
    It selects the best epoch based on validation loss and saves the weights for the best model.

    Args:
        model (torch.nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam optimizer).
        model_dir (str): Path where the best model weights will be saved.
        num_epochs (int, optional): Maximum number of epochs to train. Default is 50.
        patience (int, optional): Number of epochs with no improvement after which training will stop. Default is 5.
        device (str, optional): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        int: The epoch with the best validation performance (lowest validation loss).
    """
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_epoch = 0
    best_valid_loss = float('inf')  # Using infinity instead of 1e9 for clarity
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        print(f"Epoch {epoch+1} \n-------------------------------")

        # Training loop
        for batch, (X, Y) in tqdm.tqdm(enumerate(train_loader)):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        valid_loss = 0.0
        
        with torch.no_grad():
            for batch, (X, Y) in tqdm.tqdm(enumerate(valid_loader)):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                loss = criterion(pred, Y)
                valid_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        # Print losses
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")

        # Check for early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1  # Storing the best epoch (1-indexed)
            patience_counter = 0  
            # Save the best model's weights
            torch.save(model.state_dict(), f"{model_dir}/best_m2encoder_model.pt")
        else:
            patience_counter += 1

        # If early stopping is triggered, break the loop
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_epoch


def admet_performance(seed_options, sID, device, lr, batch_size, num_epochs, 
                      group, group_config, model_dir, twolayers, seq_len, symbol_ID):
    """
    This function loops through the ADMET TDC datasets, prepares the data, model, loss functions, and optimizer for training.
    It trains the model with early stopping and evaluates it using the test split with the best model weights.

    Args:
        seed_options (list): List of seeds for reproducibility in train and validation splits.
        sID (int): Vocabulary size.
        device (str): Device for computation - 'cuda' or 'cpu'.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Total number of epochs for training.
        group (tdc.benchmark_group.admet_group.admet_group): ADMET TDC benchmark group.
        group_config (dict): A dictionary where the key is the benchmark name and value is the task type 
        ('classification' or 'regression').
        model_dir (str): Directory to save the best model weights.
        twolayers (bool): If True, the last two encoder layers will be activated for training.
        seq_len (int): Maximum length of SMILES input token sequence.
        symbol_ID (dict): Vocabulary.

    Returns:
        dict: A dictionary where the key is the benchmark name and the value is a list of performance 
        metrics [mean, std] for that benchmark.
    """
    
    predictions_list = []
    
    # Loop over different seed options
    for seed in seed_options:
        start_time = time.time()
        predictions = {}
        
        # Loop over each benchmark in the ADMET group
        for benchmark in group:
            
            # Data preparation
            data_name = benchmark['name']
            task_type = group_config[data_name]
            train_val, test = benchmark['train_val'], benchmark['test']
            train, valid = group.get_train_valid_split(benchmark=data_name, split_type='default', seed=seed)
            mean = train['Y'].mean()
            sd = train['Y'].std()

            # Dataset initialization
            train_dataset = SMILESDataset2enc(data=train, seq_len = seq_len, task_type=task_type, 
                                              symbol_ID = symbol_ID, mean=mean, sd=sd, 
                                              max_length_symbol = 5, change_smi = True, BOS_ID = 2, 
                                              EOS_ID = 3, PAD_ID = 0)
            valid_dataset = SMILESDataset2enc(data=valid, seq_len = seq_len, task_type=task_type, 
                                              symbol_ID = symbol_ID, mean=mean, sd=sd, 
                                              max_length_symbol = 5, change_smi = False, BOS_ID = 2, 
                                              EOS_ID = 3, PAD_ID = 0)
            test_dataset = SMILESDataset2enc(data=test, seq_len = seq_len, task_type=task_type, 
                                              symbol_ID = symbol_ID, mean=mean, sd=sd, 
                                              max_length_symbol = 5, change_smi = False, BOS_ID = 2, 
                                              EOS_ID = 3, PAD_ID = 0)
    
            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=10)
            valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=1, num_workers=10)
            test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, num_workers=10)

            # Model configuration
            model = Model_2enc(lID = 1, task_type = task_type, NumToken = sID).to('cuda')

            model.smiles_encoder.load_state_dict(torch.load("/vol/bertdata/2encoder/OdorCode-40 smiles_encoder D256.Hidden512.Head8.L10.R0.5.S100000-epoch.600"))
            model.symbol_encoder.load_state_dict(torch.load("/vol/bertdata/2encoder/OdorCode-40 symbol_encoder D256.Hidden512.Head8.L10.R0.5.S100000-epoch.600"))

            for param in model.symbol_encoder.parameters():
                param.requires_grad = False

            for param in model.smiles_encoder.parameters():
                param.requires_grad = False
            for param in model.smiles_encoder.transformer_encoder.layers[-1].parameters():
                param.requires_grad = True

            if twolayers:
                for param in model.smiles_encoder.transformer_encoder.layers[-2].parameters():
                    param.requires_grad = True
                        
            model.to(device)

            # Loss function and optimizer
            if task_type == 'classification':
                criterion = nn.BCELoss()
            else:
                criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train the model with early stopping
            best_epoch = train_model(model=model, train_loader=train_dataloader, 
                                     valid_loader=valid_dataloader, criterion=criterion, 
                                     optimizer=optimizer, model_dir=model_dir, 
                                     num_epochs=num_epochs, patience=10, device=device)

            # Re-initialize model for evaluation with the best weights
            model.load_state_dict(torch.load(f"{model_dir}/best_m2encoder_model.pt", weights_only=True))
            model.eval()
            y_pred_test = []
            
            # Evaluate on test set
            with torch.no_grad():
                for batch, (X, Y) in tqdm.tqdm(enumerate(test_dataloader)):
                    X, Y = X.to(device), Y.to(device)
                    pred = model(X)
            
                    # Post-process prediction
                    if task_type == 'classification':
                        pred_save = pred.item()
                    else:
                        pred_save = (pred.item() * sd) + mean
                        
                    y_pred_test.append(pred_save)
            
            predictions[data_name] = y_pred_test
        
        predictions_list.append(predictions)
        elapsed_time = time.time() - start_time
        print(f"For seed {seed}, the computation time is {elapsed_time:.4f} seconds\n")
    
    return group.evaluate_many(predictions_list)




