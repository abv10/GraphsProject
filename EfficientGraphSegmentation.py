import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from felzenszwalb_segmentation import segment
import torch
import EvaluationHelpers as eval
import time

if __name__ == "__main__":
    with open('prostate_test.txt', 'r') as f:
    # Read all lines in the file into a list
        lines = f.readlines()


    IOU_sum = 0
    DICE_sum = 0
    recall_sum = 0
    precision_sum = 0
    time_sum = 0

    count = 0

    for line in lines:
        count += 1
        image = Image.open('ProstateX/Processed/' + line.strip())
        mask = Image.open("ProstateX/Masks/" + line.strip())
    
        image = image.convert("RGB")
        image = np.array(image)
        mask = np.array(mask) /255

        start_time = time.time()
        segmented_image = segment(image, 0.2, 300, 100)
        segmented_image = segmented_image.astype(np.uint8)   
        segmented_image = np.dot(segmented_image[..., :3], [0.2989, 0.5870, 0.1140])
        segmented_image = np.where(segmented_image < 85, 1, 0)
        total_time = time.time() - start_time
        #Image.fromarray((segmented_image * 255).astype(np.uint8)).save("EGBS_Output/" + line.strip())

        segmented_image = segmented_image.reshape(1, segmented_image.shape[0], segmented_image.shape[1])

        
        mask = mask.reshape(1, mask.shape[0], mask.shape[1])

        IOU_sum += eval.IOU(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True)
        DICE_sum += eval.DICE(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True)
        recall_sum += eval.recall(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True)
        precision_sum += eval.precision(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True)
        time_sum += total_time

        if count % 10 == 0:
            print("IOU", IOU_sum/count)
            print("DICE", DICE_sum/count)
            print("recall", recall_sum/count)
            print("precision", precision_sum/count)
            print("time", time_sum/count)


