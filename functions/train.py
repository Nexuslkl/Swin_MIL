import torch
import torch.nn as nn
import math
from utils import gm
import torchvision


def train(path_work, model, dataloader_train, device, hp, valid_fn=None, dataloader_valid=None, test_num_pos=0):
    if hp['pretrain'] is True:
        model = model.pretrain(model, device)
    
    r = hp['r']
    lr = hp['lr']
    wd = hp['wd']
    num_epoch = hp['epoch']
    best_result = 0
    print('Learning Rate: ', lr)
    loss_fn = nn.BCELoss()
    dataset_size = len(dataloader_train.dataset)

    if hp['optimizer'] == 'side':
        params1 = list(map(id, model.decoder1.parameters()))
        params2 = list(map(id, model.decoder2.parameters()))
        params3 = list(map(id, model.decoder3.parameters()))
        base_params = filter(lambda p: id(p) not in params1 + params2 + params3, model.parameters())
        params = [{'params': base_params},
                  {'params': model.decoder1.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder2.parameters(), 'lr': lr / 100, 'weight_decay': wd},
                  {'params': model.decoder3.parameters(), 'lr': lr / 100, 'weight_decay': wd}]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    print("{:*^50}".format("training start"))
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0
        step = 0
        batch_num = len(dataloader_train)

        for index, batch in enumerate(dataloader_train):
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            
            side1, side2, side3, fusion = model(image)
            loss1 = loss_fn(gm(side1, r).to(device), label)
            loss2 = loss_fn(gm(side2, r).to(device), label)
            loss3 = loss_fn(gm(side3, r).to(device), label)
            lossf = loss_fn(gm(fusion, r).to(device), label)
            loss = loss1 + loss2 + loss3 + lossf

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1
            
            if index % 10 == 0:
                print("batch %d/%d loss:%0.4f" % (index, batch_num, loss.item()))
        epochs = epoch + 1
        average_loss = epoch_loss / math.ceil(dataset_size // dataloader_train.batch_size)
        print("epoch %d loss:%0.4f" % (epochs, average_loss))

        if valid_fn is not None:
            model.eval()
            result = valid_fn(model, dataloader_valid, test_num_pos, device)
            print('epoch %d loss:%.4f result:%.3f' % (epochs, average_loss, result))
            if result > best_result:
                best_result = result
                torch.save(model.state_dict(), path_work + 'best_model.pth')
        torch.save(model.state_dict(), path_work + 'final_model.pth')

    print('best result: %.3f' % best_result)

