import torch
import time
import os
import csv


class Trainer:

    def __init__(self, nn_model, criterion, optimizer, save_path='model/'):
        self.nn_model = nn_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_path = save_path

    def train(self, train_data_loader, num_epochs, log_step=10, timestamp='',
              start_epoch=0):
        train_loss = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for epoch in range(start_epoch, num_epochs, 1):
            running_loss = 0.0

            batch_num = 0
            tic = time.perf_counter()
            # iteration over mini-batches
            for train_data in train_data_loader:
                # retrieve input training image
                train_input_img, train_target_img = train_data
                # move data to gpu if possible
                train_input_img, train_target_img = \
                    train_input_img.to(device, non_blocking=True), \
                    train_target_img.to(device, non_blocking=True)
                # set gradients to zero at the beginning of each mini-batch
                self.optimizer.zero_grad()
                # forward propagation
                _, output = self.nn_model(train_input_img)
                # compute loss at the output
                loss = self.criterion(output, train_target_img)
                # backward propagation (compute gradients)
                loss.backward()
                # update weights
                self.optimizer.step()
                # running_loss += loss.item()
                running_loss += loss.detach()
                batch_num += 1
                # print('test: ', batch_num, end='\r', flush=True)
                print('batch number: ', batch_num, '/', len(train_data_loader),
                      ' >>> loss: ', loss.detach(), end='\r', flush=True)

            loss = running_loss.item() / len(train_data_loader)
            train_loss.append([epoch, loss])

            if (epoch % log_step) == 0:
                self.save_checkpoint(epoch, train_loss, timestamp)

            toc = time.perf_counter()
            print('Epoch: ', epoch+1, ' train loss:', loss, ' time: ', toc-tic)

        return train_loss

    def eval_model(self, eval_data_loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        running_loss = 0.0
        for eval_data in eval_data_loader:
            # retrieve input training image
            input_img, target_img = eval_data
            # move data to gpu if possible
            input_img, target_img = \
                input_img.to(device), target_img.to(device)
            # forward propagation
            _, output = self.nn_model(input_img)
            # compute loss at the output
            loss = self.criterion(output, target_img)
            # update loss
            running_loss += loss.item()

        loss = running_loss / len(eval_data_loader)

        return loss

    def save_checkpoint(self, epoch, loss, timestamp):
        # delete old checkpoint(s)
        fnames = os.listdir(self.save_path)
        for f in fnames:
            if f.startswith('model_' + timestamp) \
                    or f.startswith('loss_' + timestamp):
                os.remove(self.save_path + f)
        epoch_string = "{:0>4}".format(epoch)

        # save new checkpoint
        torch.save({
            'model_state_dict': self.nn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss[-1]
        }, self.save_path + 'model_' + timestamp + '_epoch'
           + epoch_string + '.pt')

        # save loss list
        with open(self.save_path + '/' + 'loss_' +
                  timestamp + '_epoch' + epoch_string + '.csv', 'w') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerows(loss)