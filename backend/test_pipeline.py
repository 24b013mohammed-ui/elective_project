import logging
logging.basicConfig(level=logging.INFO)
import config
from backend_api import state

def run_test():
    try:
        import torch
        from data_preparation import fetch_and_prepare_data
        from dataset_preparation import (
            create_windowed_dataset, train_val_test_split, create_pytorch_dataloaders
        )
        from model import FinancialSpectrogramCNN
        from training import train_model
        from evaluation import evaluate_model
        import traceback

        config_overrides = {}
        data_dict = fetch_and_prepare_data(
            tickers=config_overrides.get('tickers', config.TICKERS),
            start_date=config_overrides.get('start_date', config.START_DATE),
            end_date=config_overrides.get('end_date', config.END_DATE)
        )
        
        dataset = create_windowed_dataset(
            data_dict=data_dict,
            window_length=config_overrides.get('window_length', config.WINDOW_LENGTH),
            hop_size=config_overrides.get('hop_size', config.HOP_SIZE),
            prediction_horizon=config_overrides.get('prediction_horizon', config.PREDICTION_HORIZON)
        )
        print("Phase 2 done.")
        
        split_data = train_val_test_split(
            dataset=dataset,
            train_ratio=config_overrides.get('train_ratio', config.TRAIN_RATIO),
            val_ratio=config_overrides.get('val_ratio', config.VAL_RATIO),
            test_ratio=config_overrides.get('test_ratio', config.TEST_RATIO)
        )
        
        dataloaders = create_pytorch_dataloaders(
            split_data=split_data,
            batch_size=config_overrides.get('batch_size', config.BATCH_SIZE),
            shuffle_train=False
        )
        print("Phase 3 done.")
        
        model = FinancialSpectrogramCNN(num_features=5)
        device = config_overrides.get('device', config.DEVICE)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config_overrides.get('learning_rate', config.LEARNING_RATE))
        loss_fn = torch.nn.MSELoss()
        
        training_history = train_model(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=config_overrides.get('num_epochs', config.NUM_EPOCHS),
            device=device,
            checkpoint_dir='checkpoints/',
            early_stopping_patience=1
        )
        print("Phase 4 done.")
        
        checkpoint = torch.load('checkpoints/best_model.pt')
        print("Loaded checkpoint done.")
    except Exception as e:
        print("EXCEPTION RAISED:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
