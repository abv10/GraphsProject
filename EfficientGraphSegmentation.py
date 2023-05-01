import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from felzenszwalb_segmentation import segment
import torch
import EvaluationHelpers as eval
import time
import sys
from multiprocessing import Pool, cpu_count

def process_line(line, sigma, k, min_size, cutoff, dataset_path):

    image = Image.open(dataset_path + '/Processed/' + line.strip())
    mask = Image.open(dataset_path + "/Masks/" + line.strip())
    image = image.convert("RGB")
    image = np.array(image)
    mask = np.array(mask) /255
    start_time = time.time()
    segmented_image = segment(image, sigma=sigma, k=k, min_size=min_size)
    segmented_image = segmented_image.astype(np.uint8)
    segmented_image = np.dot(segmented_image[..., :3], [0.2989, 0.5870, 0.1140])
    if "LITS" in dataset_path:
        segmented_image = np.where(segmented_image < cutoff, 2,
                np.where(segmented_image < cutoff*2, 1, 0))
    else:
        segmented_image = np.where(segmented_image < cutoff, 1, 0)
    total_time = time.time() - start_time
    segmented_image = segmented_image.reshape(1, segmented_image.shape[0], segmented_image.shape[1])
    mask = mask.reshape(1, mask.shape[0], mask.shape[1])
    return (eval.IOU(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True),
            eval.DICE(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True),
            eval.recall(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True),
            eval.precision(y_target=torch.tensor(mask), y_predict=torch.tensor(segmented_image), reduce=True),
            total_time)

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
    args = sys.argv[1:]

    # Assign the first 4 arguments to variables
    data_set_path = args[0]

    sigma = float(args[1])
    k = int(args[2])
    min_size = int(args[3])
    cutoff = int(args[4])

    print("Dataset:", data_set_path, "Sigma:", sigma, "k:",k,"min_size:", min_size, "cutoff:", cutoff)

    # Define the number of processes to use (should not exceed number of CPU cores)
    num_processes = min(cpu_count(), len(lines))

    start = time.time()
    with Pool(num_processes) as p:
        results = p.starmap(process_line, [(line, sigma, k, min_size, cutoff) for line in lines[0:100]])

    for i in range(len(results)):
        IOU_sum += results[i][0]
        DICE_sum += results[i][1]
        recall_sum += results[i][2]
        precision_sum += results[i][3]
        time_sum += results[i][4]

    print("TOTAL TIME:", time.time()-start)

    count = len(lines[0:100])

    print("COUNT:", count)
    print("IOU", IOU_sum/count)
    print("DICE", DICE_sum/count)
    print("recall", recall_sum/count)
    print("precision", precision_sum/count)
    print("time", time_sum/count)
