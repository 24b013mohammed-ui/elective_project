import logging
logging.basicConfig(level=logging.INFO)
import config
import traceback

try:
    from data_preparation import fetch_and_prepare_data
    from dataset_preparation import create_windowed_dataset
    
    print("Starting data fetch...")
    config_overrides = {}
    data_dict = fetch_and_prepare_data(
        tickers=config_overrides.get('tickers', config.TICKERS),
        start_date=config_overrides.get('start_date', config.START_DATE),
        end_date=config_overrides.get('end_date', config.END_DATE)
    )
    print('Phase 1 done')
    
    dataset = create_windowed_dataset(
        data_dict=data_dict,
        window_length=config_overrides.get('window_length', config.WINDOW_LENGTH),
        hop_size=config_overrides.get('hop_size', config.HOP_SIZE),
        prediction_horizon=config_overrides.get('prediction_horizon', config.PREDICTION_HORIZON)
    )
    print('Phase 2 done')
except Exception as e:
    with open('test_error.txt', 'w') as f:
        f.write("ERROR OCCURRED:\n")
        f.write(traceback.format_exc())
    print("Error written to test_error.txt")
