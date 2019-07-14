import os
import math
import torch
import random
import pickle
import numpy as np
from torch import nn, optim
from datetime import datetime
from collections import Counter
from utils.data import preprocess
from models.scoring_functions import RNNScorer
from models.config import model_config as conf


emotion_dict = {'bad': 0, 'good': 1}

device = 'cuda:{}'.format(config['gpu']) if \
         torch.cuda.is_available() else 'cpu'

model = RNNScorer(config)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

train_batches = load_data()
test_pairs = load_data(test=True)

best_acc = 0
for epoch in range(config['n_epochs']):
    losses = []
    for batch in train_batches:
        inputs = batch[0].unsqueeze(0)  # frame in format as expected by model
        targets = batch[1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad()
        optimizer.zero_grad()

        predictions = model(inputs)
        predictions = predictions.to(device)

        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # evaluate
    with torch.no_grad():
        inputs = test_pairs[0].unsqueeze(0)
        targets = test_pairs[1]

        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = torch.argmax(model(inputs), dim=1)  # take argmax to get class id
        predictions = predictions.to(device)

        # evaluate on cpu
        targets = np.array(targets.cpu())
        predictions = np.array(predictions.cpu())

        # Get results
        # plot_confusion_matrix(targets, predictions,
        #                       classes=emotion_dict.keys())
        performance = evaluate(targets, predictions)
        if performance['acc'] > best_acc:
            print(performance)
            best_acc = performance['acc']
            # save model and results
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, 'runs/{}-best_model.pth'.format(config['model_code']))

            with open('results/{}-best_performance.pkl'.format(config['model_code']), 'wb') as f:
                pickle.dump(performance, f)
