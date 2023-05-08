import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import networkx
import torch_geometric.nn as pyg_nn
import torch_geometric as pyg
import torch.nn.functional as F
import torch_scatter
import itertools
import pickle
import time


def dice_coefficient(ground_truth, prediction):
    intersection = np.sum(ground_truth * prediction)
    gt_sum = np.sum(ground_truth)
    pred_sum = np.sum(prediction)

    dice = (2* intersection) / (gt_sum + pred_sum + 1)
    return dice


def iou_coefficient(ground_truth, prediction):
    intersection = np.sum(ground_truth * prediction)
    union = np.sum(ground_truth) + np.sum(prediction) - intersection

    iou = intersection / (union + 1)

    return iou


def recall_coefficient(ground_truth, prediction):
    true_positive = np.sum(ground_truth * prediction)
    false_negative = np.sum(ground_truth * (1 - prediction))

    recall = true_positive / (true_positive + false_negative + 1)
    return recall


def precision_coefficient(ground_truth, prediction):
    true_positive = np.sum(ground_truth * prediction)
    false_positive = np.sum((1 - ground_truth) * prediction)

    precision = true_positive / (true_positive + false_positive + 1)
    return precision

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


# print("Torch version", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
IOU_sum = []
DICE_sum = []
recall_sum = []
precision_sum = []
time_sum = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = CustomCNN()
# model1.load_state_dict(torch.load("C:/Users/Sam/Documents/GitHub/GraphsProject/cnn_weights.pth"))
model1.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 3e-3
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
in_channels = 100 # Number of input features
hidden_channels = 8  # Number of hidden units
out_channels = 3  # Number of classes
attention_layer = CustomAttentionLayer(in_channels)
# attention_layer.load_state_dict(torch.load("C:/Users/Sam/Documents/GitHub/GraphsProject/att_weights.pth"))
attention_layer.to(device)
model2 = TwoLayerGCN(in_channels, hidden_channels, out_channels)
# model2.load_state_dict(torch.load("C:/Users/Sam/Documents/GitHub/GraphsProject/agcn_weights.pth"))
model2.to(device)
optimizer2 = torch.optim.Adam(itertools.chain(attention_layer.parameters(), model2.parameters()), lr=0.001)
for i in range(0, 6):
    num_epochs = 1
    for epoch in range(num_epochs):
        print(epoch)
        if i < 5:
            file_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/prostate_fold_" + str(i) + ".txt"
            data_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_fold_" + str(
                i) + ".pkl"
            mask_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_maskfold_" + str(
                i) + ".pkl"
        else:
            file_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/prostate_test.txt"
            data_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_fold_test.pkl"
            mask_output_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/AGCN superpixels/pros_maskfold_test.pkl"
        with open(file_path, 'r') as file, open(data_output_path, 'rb') as input, open(mask_output_path, 'rb') as maskput:
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
                num_segments = 800
                segments = slic(image, n_segments=num_segments, compactness=150, sigma=1)
                graph = create_superpixel_graph(segments)
                # marked_image = mark_boundaries(image, segments)
                # marked_mask = mark_boundaries(mask, segments)
                all_patches = pickle.load(input)
                patch_labels = pickle.load(maskput)
                dataset = SuperpixelPatchDataset(all_patches, patch_labels)
                train_size = int(len(dataset))

                train_dataset = dataset
                batch_size = 32

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                random_noise_scale = 0.1
                if i == 0:
                    model1.train()
                    for batch_idx, (data, targets) in enumerate(train_loader):
                        data = data.to(device)
                        targets = targets.to(device)
                        optimizer1.zero_grad()
                        output = model1(data)
                        loss = criterion(output, targets)
                        loss.backward()
                        for param in model1.parameters():
                            param.grad.data.add_(torch.randn_like(param.grad) * random_noise_scale)
                        optimizer1.step()
                else:
                    features = extract_features(all_patches, model1, device)

                    add_features_to_graph(graph, features)
                    add_labels_to_graph(graph, patch_labels)

                    data = nx_to_pyg_data(graph)
                    targets = data.y.to(device)
                    node_features = data.x.clone().detach()
                    normalized_laplacian = compute_normalized_laplacian(graph)

                    # Training loop
                    if i < 5:

                        model2.train()
                        # Compute the attention coefficients
                        edge_index = data.edge_index.to(device)
                        node_features = node_features.to(device)
                        attention_tensor = attention_layer(node_features, edge_index).cpu()
                        attention_coefficients = attention_tensor.detach().numpy()
                        flattened_attention_coefficients = attention_coefficients.flatten()
                        # Create the adjacency matrix using the attention coefficients
                        attention_adj_matrix = attention_coefficients_to_adj_matrix(
                            data.edge_index, flattened_attention_coefficients, len(graph.nodes))

                        # Compute the Hadamard product
                        hadamard_product = np.multiply(normalized_laplacian, attention_adj_matrix)
                        hadamard_product_tensor = torch.tensor(hadamard_product, dtype=torch.float32)
                        hadamard_product_tensor = hadamard_product_tensor.to(device)

                        # Forward pass
                        output = model2(hadamard_product_tensor, node_features)
                        loss = criterion(output, targets)

                        # Backward pass
                        optimizer2.zero_grad()
                        loss.backward()
                        optimizer2.step()
                        # Print the loss for this epoch
                        # print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
                    else:
                        attention_layer.eval()
                        model2.eval()
                        correct = 0
                        total = 0
                        start_time = time.time()
                        with torch.no_grad():
                            edge_index = data.edge_index.to(device)
                            node_features = node_features.to(device)
                            attention_tensor = attention_layer(node_features, edge_index).cpu()
                            attention_coefficients = attention_tensor.detach().numpy()
                            flattened_attention_coefficients = attention_coefficients.flatten()
                            # Create the adjacency matrix using the attention coefficients
                            attention_adj_matrix = attention_coefficients_to_adj_matrix(
                                data.edge_index, flattened_attention_coefficients, len(graph.nodes))

                            # Compute the Hadamard product
                            hadamard_product = np.multiply(normalized_laplacian, attention_adj_matrix)
                            hadamard_product_tensor = torch.tensor(hadamard_product, dtype=torch.float32)
                            hadamard_product_tensor = hadamard_product_tensor.to(device)
                            # Forward pass
                            output = model2(hadamard_product_tensor, node_features)
                            total_time = time.time() - start_time
                            _, predicted = torch.max(output.data, 1)
                            IOU_sum.append(iou_coefficient(targets.cpu().numpy(), predicted.cpu().numpy()))
                            DICE_sum.append(dice_coefficient(targets.cpu().numpy(), predicted.cpu().numpy()))
                            recall_sum.append(recall_coefficient(targets.cpu().numpy(), predicted.cpu().numpy()))
                            precision_sum.append(precision_coefficient(targets.cpu().numpy(), predicted.cpu().numpy()))
                            time_sum.append(total_time)

    if i == 0:
        model_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/cnn_weights.pth"
        torch.save(model1.state_dict(), model_path)
    elif i == 4:
        model_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/att_weights.pth"
        torch.save(attention_layer.state_dict(), model_path)
        model_path = "C:/Users/Sam/Documents/GitHub/GraphsProject/agcn_weights.pth"
        torch.save(model2.state_dict(), model_path)
    elif i == 5:
        print("IOU", np.mean(IOU_sum))
        print("DICE", np.mean(DICE_sum))
        print("recall", np.mean(recall_sum))
        print("precision", np.mean(precision_sum))
        print("time", np.mean(time_sum))

# plt.imshow(marked_image)
# plt.show()
# plt.imshow(marked_mask)
# plt.show()
# cv2.imshow('Patch', patch)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# Press the green button in the gutter to run the script.
##if __name__ == '__main__':

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
