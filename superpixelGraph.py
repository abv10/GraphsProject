import cv2
import numpy as np
from skimage.segmentation import slic
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import networkx
import torch_geometric.nn as pyg_nn
import torch_geometric as pyg
import torch.nn.functional as F
import torch_scatter
import pickle
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
def create_superpixel_graph(superpixels):
    graph = networkx.Graph()

    height, width = superpixels.shape
    for y in range(height):
        for x in range(width):
            node = superpixels[y, x] - 1
            if not graph.has_node(node):
                graph.add_node(node)

            # Check for adjacent superpixels (4-connectivity)
            neighbors = [
                (y, x - 1),
                (y - 1, x),
                (y, x + 1),
                (y + 1, x)
            ]

            for ny, nx in neighbors:
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_node = superpixels[ny, nx] - 1
                    if neighbor_node != node and not graph.has_edge(node, neighbor_node):
                        graph.add_edge(node, neighbor_node)

    return graph


def attention_coefficients_to_adj_matrix(edge_index, attention_coefficients, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i, (source, target) in enumerate(edge_index.t().numpy()):
        if source >= num_nodes or target >= num_nodes:
            print(f"Invalid edge: ({source}, {target})")
            continue

        adj_matrix[source, target] = attention_coefficients[i]

    return adj_matrix


def compute_normalized_laplacian(graph):
    laplacian = networkx.normalized_laplacian_matrix(graph).toarray()
    return laplacian


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(40),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(80),
            nn.LeakyReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Sequential(
            nn.Linear(80 * 4 * 4, 100),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        return x

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        features = x.view(x.size(0), -1)
        return features


class CustomAttentionLayer(nn.Module):
    def __init__(self, in_channels, negative_slope=0.2):
        super(CustomAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.negative_slope = negative_slope

        self.att = nn.Parameter(torch.Tensor(2 * in_channels, 1))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        row, col = edge_index

        x = torch.cat([x[row], x[col]], dim=-1)
        alpha = torch.matmul(x, self.att)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha_softmax = torch_scatter.scatter_softmax(alpha.t(), row)

        return alpha_softmax.squeeze(-1)


class CustomGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, hadamard_product, features):
        out = torch.matmul(hadamard_product, features)
        return self.linear(out)


class TwoLayerGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TwoLayerGCN, self).__init__()
        self.gcn1 = CustomGCNLayer(in_channels, hidden_channels)
        self.gcn2 = CustomGCNLayer(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hadamard_product, features):
        x = self.relu(self.gcn1(hadamard_product, features))
        x = self.gcn2(hadamard_product, x)
        return self.softmax(x)


def nx_to_pyg_data(graph):
    x = torch.tensor(np.stack([graph.nodes[node]["feature"] for node in graph.nodes]), dtype=torch.float)
    y = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes], dtype=torch.long)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()

    return pyg.data.Data(x=x, edge_index=edge_index, y=y)


def place_superpixel_into_patch(image, segments, superpixel_id, patch_width, patch_height):
    coords = np.where(segments == superpixel_id)
    values = ground_truth_array[coords]
    counts = np.bincount(values)
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])

    # superpixel_content = image[y_min:y_max + 1, x_min:x_max + 1, :]
    superpixel_height, superpixel_width = y_max - y_min + 1, x_max - x_min + 1
    y_offset = (patch_height - superpixel_height) // 2
    x_offset = (patch_width - superpixel_width) // 2
    patch = np.zeros((patch_width, patch_height, 3), dtype=np.uint8)

    for y, x in zip(coords[0], coords[1]):
        patch_y, patch_x = y - y_min + y_offset, x - x_min + x_offset
        patch[patch_y, patch_x] = image[y, x]
    return patch, counts


def classify_pixel(pixel):
    red_threshold = 0
    return 1 if pixel[0] > red_threshold else 0


def extract_features(patches, model, device):
    model.eval()
    model.to(device)

    features = {}
    with torch.no_grad():
        for patch_id, patch in patches.items():
            # Convert the NumPy array to a PyTorch tensor
            patch = torch.from_numpy(patch).float()

            # Move the patch tensor to the specified device
            patch = patch.to(device)

            # Run the model to get the feature vector
            feature_vector = model.extract_features(patch.unsqueeze(0))

            # Save the feature vector as a NumPy array
            features[patch_id] = feature_vector.cpu().numpy().flatten()

    return features


def add_features_to_graph(graph, features):
    for node, feature in features.items():
        graph.nodes[node]["feature"] = feature


def add_labels_to_graph(graph, labels):
    for node, label in labels.items():
        graph.nodes[node]["label"] = label


class SuperpixelPatchDataset(Dataset):
    def __init__(self, patches, labels, transform=None):
        self.patches = patches
        self.labels = labels
        self.transform = transform
        self.keys = list(patches.keys())  # Assuming both dictionaries have the same keys

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        key = self.keys[idx]  # Get the key for the current index
        patch = self.patches[key]
        label = self.labels[key]

        if self.transform:
            patch = self.transform(patch)

        return patch, label


for i in range(0, 6):
    if i < 5:
        file_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/prostate_fold_" + str(i) + ".txt"
        data_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_fold_" + str(i) + ".pkl"
        mask_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_maskfold_" + str(
            i) + ".pkl"
    else:
        file_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/prostate_test.txt"
        data_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_fold_test.pkl"
        mask_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_maskfold_test.pkl"
    with open(file_path, 'r') as file, open(data_output_path, "wb") as output, open(mask_output_path, "wb") as maskput:
        # Iterate through each line in the file
        for line in file:
            # Remove any leading and trailing whitespace (including newline characters)
            line = line.strip()
            image_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/ProstateX/Processed/" + line
            mask_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/ProstateX/Masks/" + line
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_height, mask_width, _ = mask.shape
            ground_truth_array = np.zeros((mask_height, mask_width), dtype=np.float32)
            for y in range(mask_height):
                for x in range(mask_width):
                    ground_truth_array[y, x] = classify_pixel(mask[y, x])
            if np.any(ground_truth_array == .5):
                ground_truth_array = ground_truth_array * 2
            ground_truth_array = ground_truth_array.astype(np.uint8)
            num_segments = 800
            segments = slic(image, n_segments=num_segments, compactness=120, sigma=1)
            # graph = create_superpixel_graph(segments)
            marked_image = mark_boundaries(image, segments)
            # marked_mask = mark_boundaries(mask, segments)
            all_patches = {}
            patch_labels = {}
            for superpixel_id in np.unique(segments):
                patch, counts = place_superpixel_into_patch(image, segments, superpixel_id, 32, 32)
                patch = np.array(patch, dtype=np.float32)
                patch = np.transpose(patch, (2, 0, 1))  # Convert to PyTorch format (C, H, W)
                # patch = torch.tensor(patch).unsqueeze(0)  # Add batch dimension
                all_patches[superpixel_id - 1] = patch
                patch_labels[superpixel_id - 1] = np.argmax(counts)
            pickle.dump(all_patches, output)
            pickle.dump(patch_labels, maskput)
