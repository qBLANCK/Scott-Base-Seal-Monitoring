import time
import copy
import torch
import numpy as np
from sklearn.metrics import f1_score

from Models.Snowstorm.constants import FEATURE_EXTRACT, NUM_CLASSES


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
    since = time.time()

    score_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0
    best_score = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348/2
            f1_labels = []
            f1_preds = []

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # Calculate f1-score
                f1_preds.append(preds.cpu())
                f1_labels.append(labels.cpu().data)

            f1_preds = np.concatenate(f1_preds)
            f1_labels = np.concatenate(f1_labels)
            # epoch_f1 = f1_score(labels.cpu().data, preds.cpu())
            epoch_f1 = f1_score(f1_labels, f1_preds)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_f1))

            # deep copy the model
            if phase == 'val':
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                # Score to optimise
                # score = epoch_acc
                score = epoch_f1
                if score > best_score:
                    best_score = score
                    best_model_wts = copy.deepcopy(model.state_dict())
                score_history.append((best_score, best_f1))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, score_history

