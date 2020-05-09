from models.config import model_config
from utils.preprocess import process_raw, create_train_val_split
from utils.generate import run

if __name__ == '__main__':
    # data = run(model_config)
    dataset = process_raw(model_config)
    # since spliting is happening in process_bimodal iteself
    if model_config['model_code'] not in ['bimodal_scorer', 'spec_clf', 'bvae']:
        create_train_val_split(dataset, model_config)
