import EGISsource.segmentation as seg    #import library
import EGISsource.disjoint_set
from skimage import transform
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')

def felzenszwalb_segmentation(image_path, scale, min_size, num_classes):
    # Step 1
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img)
    image = img / 255.0
    
    grid_graphs = seg.generate_graph(image)
    Segs = seg.segmentation(grid_graphs, image.shape[0], image.shape[1], k=1)
    segmented_img= seg.draw_img(Segs, image.shape[0], image.shape[1])
    segmented_img = segmented_img * 255
    segmented_img = segmented_img.astype(np.uint8)
    pil_image = Image.fromarray(segmented_img)

# Show the image
    pil_image.show()
    print(segmented_img.shape)


if __name__ == "__main__":
    image = Image.open('ProstateX/Processed/p0000slice9.png')
    image_data = np.asarray(image)
    #image.show()
    felzenszwalb_segmentation('ProstateX/Processed/p0000slice9.png',0.15*image_data.shape[0], 20, 12)
