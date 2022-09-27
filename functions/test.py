import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import hausdorff, dice_coeff, gm

def test(path_work, model, dataloader, device, weight=None):
    if weight is None:
        path_model = path_work + 'best_model.pth'
    elif weight == 'final':
        path_model = path_work + 'final_model.pth'
    else:
        path_model = path_work + weight
    model.load_state_dict(torch.load(path_model))
    model.eval()
    plt.ion()

    step = 0
    total_f = 0
    total_hd = 0
    total_time = 0
    num = 0
    split = 80
    num_1 = 0
    num_0 = 0
    total_f_1 = 0
    total_f_0 = 0
    f1 = 0
    loss = 0
    max = 0
    mask_gm = 0
    gm = 0

    with torch.no_grad():
        for image, label, image_show in dataloader: #
            # time_start = time.time()
            step += 1

            preds = model(image.to(device))

            pred = ((preds[3] >= 0.5) + 0).squeeze(0).squeeze(0).to('cpu').numpy()
            label = label.squeeze(0).squeeze(0).int().to("cpu").numpy()
            image_show = image_show.squeeze(0).int().to("cpu").numpy()

            if step <= split:
                num_1 += 1
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
                total_f_1 += f1
                average_f_1 = total_f_1 / num_1
                hausdorff_distance = hausdorff(pred, label)
                total_hd += hausdorff_distance
                average_hd = total_hd / num
            else:
                num_0 += 1
                f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)
                total_f_0 += f1
                average_f_0 = total_f_0 / num_0
        print('F1 Pos = %.3f' % average_f_1)
        print("average HD = %.3f" % average_hd)
        print('F1 Neg = %.3f' % average_f_0)

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(image_show)
            # plt.title("%dth F = %.3f" % (step, f1))
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(1, 3, 2)
            # plt.imshow(label, cmap='YlOrRd')
            # plt.title("Ground truth")
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(1, 3, 3)
            # plt.imshow(pred, cmap='YlOrRd')
            # plt.title("Prediction")
            # plt.xticks([])
            # plt.yticks([])
            # plt.pause(1)
            # plt.savefig("output/unet_idt/com/%04d.jpg" % step)
            # plt.show()
            # plt.imsave('output/unet_idt/vis/%04d.jpg' % step, pred, cmap='YlOrRd')

        #     if num == 1:
        #         continue
        #
        #     time_end = time.time()
        #
        #     running_time = time_end - time_start
        #     total_time += running_time
        #     average_running_time = total_time / (num - 1)
        #     print('Running Time: ', running_time)
        # print('Average Time: ', average_running_time)
