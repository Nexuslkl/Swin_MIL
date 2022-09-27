import torch
import numpy as np
from sklearn import metrics


def valid(model, dataloader, test_num_pos, device):
    step = 0
    num = 0
    total_f = 0

    with torch.no_grad():
        for image, label in dataloader:
            step += 1
            if step > test_num_pos: 
                break
            num += 1
            
            preds = model(image.to(device))
            pred = ((preds[3] >= 0.5) + 0).squeeze(0).squeeze(0).to('cpu').numpy()
            label = label.squeeze(0).squeeze(0).int().to("cpu").numpy()

            temp1 = label.sum().item()
            if temp1 != 0:
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
            else:
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)

            total_f += f1
        average_f = total_f/num

        return average_f