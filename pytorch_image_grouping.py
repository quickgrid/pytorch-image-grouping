"""A rough implementation of image clustering based on feature embedding from deep learning model.

Currently implementation should give somewhat acceptable result on classes that are well separated.
Try pairwise distance implementation in scikit with reduced features or output from fc layer.

Notes:
    - Need to speedup code.
    - Verify if actually working.
    - Improve code structure.

References:
    - https://pytorch.org/vision/master/feature_extraction.html
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    - https://newbedev.com/image-clustering-by-its-similarity-in-python
    - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    - https://github.com/quickgrid/AI-Resources/blob/master/paper-implementations/pytorch/u-net/pytorch_unet_train.py
    - https://github.com/quickgrid/AI-Resources/blob/master/paper-implementations/pytorch/wgan/pytorch_wgan_with_paper_reference.py

"""

import os
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Resnet50FeatureExtractor(nn.Module):
    def __init__(
            self,
            device
    ):
        super(Resnet50FeatureExtractor, self).__init__()

        model = resnet50(pretrained=True).to(device=device)
        model.train(mode=False)
        train_nodes, eval_nodes = get_graph_node_names(model)

        print('train_nodes')
        print(train_nodes)
        print('eval_nodes')
        print(eval_nodes)

        return_nodes = {
            # 'layer2.3.relu_2': 'layer2',
            # 'layer3.5.relu_2': 'layer3',
            'layer4.2.relu_2': 'layer4',
        }
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(
            self,
            image_dir,
            transform=None,
    ):
        super(CustomImageDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.images[item])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        data = {
            'image': image,
            'path': image_path
        }

        return data


def image_feature_extract(
    image_height,
    image_width,
    batch_size,
    device,
    image_dir
):
    image_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0]
        ),
    ])

    custom_image_dataset = CustomImageDataset(
        image_dir,
        transform=image_transform,
    )

    image_loader = DataLoader(
        custom_image_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    fx = Resnet50FeatureExtractor(device=device)

    image_path_list = []
    image_feature_list = []

    for batch_idx, (data) in enumerate(image_loader):
        image_path = data['path']
        image_path_list.append(image_path)

        img = data['image']
        extracted_feature = fx(img.to(device=device))

        # print(image_path)
        for i in extracted_feature.keys():
            img_feature = extracted_feature[i].flatten().cpu().detach().numpy()
            image_feature_list.append(img_feature)
            # print(img_feature.shape)

    print(len(image_feature_list))
    print('Done feature extraction.')

    return (image_feature_list, image_path_list)

def cluster_images(image_feature_list, image_path_list):
    # img_feature_embedding = TSNE(
    #     n_components=2, learning_rate='auto', init='random'
    # ).fit_transform(np.array(image_feature_list))
    # print(img_feature_embedding.shape)

    image_feature_list = np.array(image_feature_list)
    print(image_feature_list.shape)

    pca = PCA(n_components=18)
    img_feature_embedding = pca.fit_transform(image_feature_list)
    print(img_feature_embedding.shape)

    # TODO: Approximate nearest neighbour on the lower dimension embedding to get similar images.
    print('STARTED CLUSTERING')
    number_clusters = 3
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(img_feature_embedding))
    print(kmeans.labels_)

    image_cluster_dict = {}
    for i, m in enumerate(kmeans.labels_):
        image_cluster_dict[f'{m}'] = []

    print('CLUSTER GROUPS')
    for i, m in enumerate(kmeans.labels_):
        image_cluster_dict[f'{m}'].append(image_path_list[i])

    # print(image_cluster_dict)
    print(json.dumps(image_cluster_dict, indent=4, separators=(',', ':')))

    # Plot some similar images
    f, axarr = plt.subplots(3,3)
    for i in image_cluster_dict.keys():
        for j in range(3):
            im = Image.open(image_cluster_dict[i][j][0])
            axarr[int(i), j].imshow(np.array(im))
            axarr[int(i), j].set_title(f'Cluster {i}')
    plt.show()


if __name__ == '__main__':
    image_height = 512
    image_width = 512
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_dir = r'scattered_images'

    cluster_images(
        *image_feature_extract(
            image_height=image_height,
            image_width=image_width,
            batch_size=batch_size,
            device=device,
            image_dir=image_dir
        )
    )
