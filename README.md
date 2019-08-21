# lyrics_generation
Generate song lyrics from music

# Steps to train the scoring functions:
* Make necessary changes in `src/models/config.py` (such as paths, hyperparameters)
* Run `python src/process_raw.py` to process the dataset. It will put all the files `data/processed/{model_code}/` folder
* Run `python src/train_scorer.py`to train the scoring functions. The type of scoring function (`bimodal_scorer/bilstm_scorer`) to be trained can be controlled from the config file.
