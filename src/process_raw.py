from models.config import model_config
from utils.preprocess import process_raw, create_train_val_split

if __name__ == '__main__':
    dataset = process_raw(model_config)
    create_train_val_split(dataset, model_config)
