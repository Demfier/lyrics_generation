# lyrics_generation
Generate song lyrics from music

# Steps to train the scoring functions:
* Make necessary changes in `src/models/config.py` (such as paths, hyperparameters)
* Run `python src/process_raw.py` to process the dataset. It will put all the files `data/processed/{model_code}/` folder
* Run `python src/train_scorer.py`to train the scoring functions. The type of scoring function (`bimodal_scorer/bilstm_scorer`) to be trained can be controlled from the config file.
* Run `python src/train_model.py`to train the desired generative model. The hyperparameters can be controlled from the config file.
* Various graphs can be accessed from the tensorboard (run `tensorboard --logdir runs` to start).
* To run the model in inference mode:
  - change `sampling_temperature` to something really low (like `5e-3`)
  - switch `first_run?` to `False`
  - change `pretrained_model` from `False` to pretrained model's path. It is of the format `model_code/model_name` (`rec/vae-1L-bilstm-40`)
  - finally, run `python src/predict.py`. Currently, it would show some randomly sampled vectors and interpolation between two randomly sampled sentenes from the validation set.
