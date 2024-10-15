# In[]
import sys 
import copy
import argparse
import time
import numpy as np  
import timm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

import utils
import datasets
import prototype as prot
from model_loading import build_model

import os 

from torch.utils.data import DataLoader
import pytorch_lightning as L
import os

from sklearn.cluster import SpectralClustering, KMeans, BisectingKMeans

import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
# In[]
def get_args_parser():
    parser = argparse.ArgumentParser(description='Prototype Prompt for Continual Learning')

    # general -------------------------------------------------------------------------------------
    parser.add_argument('--bs', default=256, type=int, help='batch-size for training')
    parser.add_argument('--numerical_order', default=True, type=utils.bool_flag, help='whether fix the class order')
    parser.add_argument('--finetune_vit', default=False, type=utils.bool_flag, help='whether allow fintune vit')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet_r', '5datasets'], help='Dataset to use')
    parser.add_argument('--loss', default='protocon', type=str, help='loss function')
    parser.add_argument('--total_nc', default=100, type=int, help='total class number for the dataset')
    parser.add_argument('--fg_nc', default=10, type=int, help='the number of classes in the first task')
    parser.add_argument('--task_num', default=9, type=int, help='the number of tasks')

    # hyper-parameters for representation generation
    parser.add_argument('--split_ratio', default=0.8, type=float, help='the proportion of samples in training set to form a validation set')
    # parser.add_argument('--use_mc_proto', default=True, type=utils.bool_flag, help='whether use multi-centroid prototype')
    # parser.add_argument('--n_clusters', default=5, type=int, help='number of centroids for each prototype')
    # parser.add_argument('--cluster_method', default='spectral', type=str, choices=['spectral', 'kmeans'], help='clustering method for generating multi-centroid prototypes')

    # hyper-parameters for NCM-based associative memory ----------------------------------------------------------------------
    parser.add_argument('--similarity', default='cosine', type=str, choices=['l1','l2','cosine'], help='way to compute similarity(distance)')

    # hyper-parameters for MLP-based associative memory ----------------------------------------------
    # parser.add_argument('--train_epochs', default=100, type=int, help='Number of epochs for training MLP-based AM')

    # Backbone ---------------------------------------------------------------------------------------
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--pretrain_method', default='dino', type=str, choices=['dino', 'mae', 'deit', '1k', '21k',], help='load weights trained with different methods')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    # Misc ---------------------------------------------------------------------------------------
    parser.add_argument('--data_path', default='../data', type=str, help='Please specify path to the folder where data is saved.')
    parser.add_argument('--output_dir', default='./exps_results/associative_memory', type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--exp_name', default='', type=str, help='experiment name.')
    parser.add_argument('--seed', default=77, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--device_ids', default=[0,1], type=list)

    return parser

def set_tasks(args):
    # set task size and class order
    if args.task_num > 0:
        # number of classes in each incremental step
        args.task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    else:
        args.task_size = 0

    if args.dataset == '5datasets': 
        # always set fixed numerical order for 5-datasets for convienince
        args.class_seq = np.array(range(args.total_nc))

    else:
        if args.numerical_order:
            args.class_seq = np.array(range(args.total_nc))
        else:
            # random permutation
            args.class_seq = np.random.permutation(range(args.total_nc))

def generate_class_features(args, feature_extractor, class_id, dataset):
    # collect features for a specific class
    with torch.no_grad():
        feature_list = []
        dataset.set_classes([class_id])
        data_loader = DataLoader(dataset=dataset,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 batch_size=args.bs,
                                 drop_last=False)
        
        for _, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            outputs = feature_extractor(imgs)
            feature_list.append(outputs)
        features = torch.cat(feature_list, dim=0).cpu()
    return features

def generate_features(args, feature_extractor, dataset):
    # collect features for all classes
    train_features_dict = {}
    valid_features_dict = {}

    with torch.no_grad():
        feature_extractor.cuda().eval()
        train_prototypes, train_prototypes_var, train_prototypes_similarity, train_all_features, train_all_labels = [], [], [], [], []
        valid_prototypes, valid_prototypes_var, valid_prototypes_similarity, valid_all_features, valid_all_labels = [], [], [], [], []

        tqdm_gen = tqdm.tqdm(range(args.total_nc))
        tqdm_gen.set_description('Generate Prototypes')
        for class_id in tqdm_gen:
            tqdm_gen.set_postfix({'prototype id':class_id})

            features = \
                generate_class_features(args, feature_extractor, class_id, dataset)
            
            dataset_size = len(features)
            indices = torch.randperm(dataset_size)
            train_size = int(dataset_size * args.split_ratio)
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:]

            train_features = features[train_indices]
            valid_features = features[valid_indices]

            train_var, train_mean = torch.var_mean(train_features, dim=0)
            valid_var, valid_mean = torch.var_mean(valid_features, dim=0)

            train_cosine_similarity = utils.cosine_similarity(train_mean.unsqueeze(0), train_features)
            valid_cosine_similarity = utils.cosine_similarity(valid_mean.unsqueeze(0), valid_features)

            train_all_features.append(train_features)
            valid_all_features.append(valid_features)

            train_all_labels.append(torch.tensor([class_id]).repeat(len(train_features)))
            valid_all_labels.append(torch.tensor([class_id]).repeat(len(valid_features)))

            train_prototypes.append(train_mean)
            valid_prototypes.append(valid_mean)

            train_prototypes_var.append(train_var)
            valid_prototypes_var.append(valid_var)

            train_prototypes_similarity.append(train_cosine_similarity)
            valid_prototypes_similarity.append(valid_cosine_similarity)

        train_prototypes = torch.stack(train_prototypes, dim=0)  # shape(num_class, feature_dim)
        train_prototypes_var = torch.stack(train_prototypes_var, dim=0)  # shape(num_class, feature_dim)
        train_prototypes_similarity = train_prototypes_similarity  # list: len(num_classes)
        train_all_features = torch.cat(train_all_features, dim=0)
        train_all_labels = torch.cat(train_all_labels, dim=0)

        train_features_dict['features'] = train_all_features
        train_features_dict['labels'] = train_all_labels
        train_features_dict['prototypes'] = train_prototypes
        train_features_dict['prototypes_var'] = train_prototypes_var
        train_features_dict['similarity'] = train_prototypes_similarity

        valid_prototypes = torch.stack(valid_prototypes, dim=0)  # shape(num_class, feature_dim)
        valid_prototypes_var = torch.stack(valid_prototypes_var, dim=0)  # shape(num_class, feature_dim)
        valid_prototypes_similarity = valid_prototypes_similarity  # list: len(num_classes)
        valid_all_features = torch.cat(valid_all_features, dim=0)
        valid_all_labels = torch.cat(valid_all_labels, dim=0)

        valid_features_dict['features'] = valid_all_features
        valid_features_dict['labels'] = valid_all_labels
        valid_features_dict['prototypes'] = valid_prototypes
        valid_features_dict['prototypes_var'] = valid_prototypes_var
        valid_features_dict['similarity'] = valid_prototypes_similarity

    print('Train prototypes size: {}'.format(train_prototypes.size()))
    return train_features_dict, valid_features_dict

def prepare_features(args):
    train_features_dir = 'AM_datasets/{}_{}_train_features_dict.pth'.format(args.pretrain_method, args.dataset)
    valid_features_dir = 'AM_datasets/{}_{}_valid_features_dict.pth'.format(args.pretrain_method, args.dataset)

    try:
        train_features_dict = torch.load(train_features_dir)
        valid_features_dict = torch.load(valid_features_dir)

    except:
        feature_extractor = build_model(args)
        
        train_dataset, gen_proto_dataset, test_dataset = datasets.prepare_dataset(args)

        train_features_dict, valid_features_dict = generate_features(args, feature_extractor, gen_proto_dataset)

        torch.save(train_features_dict, train_features_dir)
        torch.save(valid_features_dict, valid_features_dir)

    return train_features_dict, valid_features_dict

class FeatureDataset(Dataset):
    """
    A custom PyTorch Dataset subclass designed to manage a dataset of features and corresponding labels.

    Attributes:
        all_data (Tensor): A tensor containing all the feature vectors.
        all_targets (Tensor): A tensor containing all the corresponding labels.
        data (Tensor): A subset of `all_data` currently being used.
        targets (Tensor): A subset of `all_targets` corresponding to `data`.

    Methods:
        __init__(features, labels):
            Initializes the FeatureDataset with the provided features and labels.
        
        set_classes(target_classes):
            Sets the `data` and `targets` attributes to include only the samples whose labels are in `target_classes`.
        
        get_num_of_class(target):
            Returns the number of samples in the dataset that belong to the specified class `target`.
        
        get_class_chunk(target):
            Returns the start and end indices of the specified class `target` within the `data`.
        
        __getitem__(idx):
            Retrieves the sample at the specified index `idx` along with its corresponding label.
        
        __len__():
            Returns the number of samples in the current dataset (`data`).
    """
    def __init__(self, features, labels) -> None:
        """
        Initializes the FeatureDataset instance with the provided features and labels.

        Parameters:
            features (Tensor): A tensor containing the feature vectors.
            labels (Tensor): A tensor containing the corresponding labels.
        """
        super().__init__()
        self.all_data = features
        self.all_targets = labels

    def set_classes(self, target_classes: list):
        """
        Sets the `data` and `targets` attributes to include only the samples whose labels are in `target_classes`.

        Parameters:
            target_classes (list): A list of class labels to filter the dataset.
        """
        data, targets = [], []
        for label in target_classes:
            current_data = self.all_data[self.all_targets == label]
            current_targets = torch.full((current_data.shape[0],), label)
            data.append(current_data)
            targets.append(current_targets)
    
        self.data = torch.cat(data, axis=0)
        self.targets = torch.cat(targets, axis=0)

    def get_num_of_class(self, target):
        return torch.sum(torch.tensor(self.targets) == target)

    def get_class_chunk(self, target): 
        """
        get start index and end index of the target in self.data
        """
        indexes = (self.targets == target).nonzero()
        start_idx, end_idx = torch.min(indexes), torch.max(indexes)
        return start_idx, end_idx
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def __len__(self):
        return len(self.data)

# In[] 
def calculate_topk_accuracy(similarity, classes, labels, k=3):
    """
    Calculates the top-k accuracy of predictions.

    This function computes the top-k accuracy by identifying the top-k most similar classes
    for each prediction and checking if the true label is among these top-k predictions.

    Returns:
        float: The top-k accuracy as a fraction of samples where the true label was among the top-k predictions.
    """
    topk_predicted_indices = torch.topk(similarity, k=k, dim=1).indices
    topk_predicted_labels = classes[topk_predicted_indices]
    
    correct_predictions = torch.zeros_like(labels).float()
    for i in range(len(labels)):
        if labels[i] in topk_predicted_labels[i]:
            correct_predictions[i] = 1.0
    accuracy = correct_predictions.mean().item()
    
    return accuracy

"""Define NCM-based associative memory"""
def evaluate_NCM_associative_memory(classes, train_features_dict, valid_features_dict, k=3):
    """
    Evaluates the Nearest Class Mean (NCM) associative memory model on validation data.

    This function calculates the top-1 and top-k accuracies by comparing the predicted class labels
    derived from the cosine similarity between validation features and class prototypes with the true labels.

    Parameters:
        classes (list or Tensor): A list or tensor of class indices to be evaluated.
        train_features_dict (dict): A dictionary containing 'prototypes' key with the mean feature vectors per class.
        valid_features_dict (dict): A dictionary containing 'features' and 'labels' keys for validation data.
        k (int, optional): The number of top predictions to consider for top-k accuracy. Defaults to 3.

    Returns:
        tuple: A tuple containing the top-1 accuracy and the top-k accuracy as floating-point numbers.
    """
    classes = torch.tensor(classes)
    features = valid_features_dict['features']
    labels = valid_features_dict['labels']
    dataset = FeatureDataset(features, labels)
    dataset.set_classes(classes)

    features = dataset.data
    labels = dataset.targets

    prototypes = train_features_dict['prototypes'][classes]

    features = torch.nn.functional.normalize(features, dim=1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    similarity = torch.mm(features, prototypes.t())

    top1_predicted_indices = torch.argmax(similarity, dim=1)
    top1_predicted_labels = classes[top1_predicted_indices]
    top1_correct_predictions = (top1_predicted_labels == labels).float()
    top1_accuracy = top1_correct_predictions.mean().item()

    topk_accuracy = calculate_topk_accuracy(similarity, classes, labels, k=k)
    return top1_accuracy, topk_accuracy

def evaluate_multi_centroid_NCM_associative_memory(
        classes, train_features_dict, valid_features_dict,
        cluster_method='spectral', n_clusters=3, topk_retrieval=3):
    """
    Evaluates the multi-centroid NCM associative memory model on validation data.

    This function calculates the top-1 and top-k accuracies by comparing the predicted class labels
    derived from the cosine similarity between validation features and multiple centroids per class
    with the true labels. It also computes the average number of retrievals needed to identify the
    correct class.

    Parameters:
        classes (list or Tensor): A list or tensor of class indices to be evaluated.
        train_features_dict (dict): A dictionary containing 'features' and 'labels' keys for training data.
        valid_features_dict (dict): A dictionary containing 'features' and 'labels' keys for validation data.
        cluster_method (str, optional): The clustering method to use ('spectral' or 'Kmeans'). Defaults to 'spectral'.
        n_clusters (int, optional): The number of clusters (centroids) to form per class. Defaults to 3.
        topk_retrieval (int, optional): The number of top predictions to consider for top-k accuracy. Defaults to 3.

    Returns:
        tuple: A tuple containing the top-1 accuracy, the top-k accuracy, and the average number of retrievals as floating-point numbers.
    """
    
    train_features = train_features_dict['features']
    train_labels = train_features_dict['labels']
    train_dataset = FeatureDataset(train_features, train_labels)
    train_dataset.set_classes(classes)
    
    prototypes = []

    label_to_class_mapping = torch.zeros((len(classes) * n_clusters,)).int()

    for step, current_class in enumerate(classes):
        train_dataset.set_classes([current_class])
        current_features = train_dataset.data
        if cluster_method == 'spectral':
            clustering = spectral_clustering(current_features, n_clusters)
        elif cluster_method == 'Kmeans':
            clustering = Kmeans(current_features, n_clusters)
            
        for label in range(n_clusters):
            cluster_indices = clustering.labels_ == label
            cluster_features = current_features[cluster_indices]
            if not torch.is_tensor(cluster_features):
                cluster_features = torch.tensor(cluster_features)
            cluster_var, cluster_mean = torch.var_mean(cluster_features, dim=0)
            prototypes.append(cluster_mean)
            label_to_class_mapping[step*n_clusters + label] = current_class
    
    prototypes = torch.stack(prototypes, dim=0)

    valid_features = valid_features_dict['features']
    valid_labels = valid_features_dict['labels']
    valid_dataset = FeatureDataset(valid_features, valid_labels)
    valid_dataset.set_classes(classes)

    features = torch.nn.functional.normalize(valid_features, dim=1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    similarity = torch.mm(features, prototypes.t())

    topk_values, topk_indices = similarity.topk(k=topk_retrieval)
    topk_predicted_labels = label_to_class_mapping[topk_indices]

    top1_correct_predictions = torch.zeros_like(valid_labels).float()
    topk_correct_predictions = torch.zeros_like(valid_labels).float()
    retrieval_number = torch.zeros(len(valid_labels)).float()

    top1_correct_predictions = (topk_predicted_labels[:, 0] == valid_labels).float()

    for i in range(len(valid_labels)):
        if valid_labels[i] in topk_predicted_labels[i]:
            topk_correct_predictions[i] = 1.0
            retrieval_number[i] = len(torch.unique(topk_predicted_labels[i]))

    top1_average_accuracy = top1_correct_predictions.mean().item()
    topk_average_accuracy = topk_correct_predictions.mean().item()
    average_retrieval = retrieval_number.mean().item()

    return top1_average_accuracy, topk_average_accuracy, average_retrieval

def evaluate_multi_centroid_NCM_task_associative_memory(
        task_split_classes, train_features_dict, valid_features_dict,
        cluster_method='spectral', n_clusters=3, retrieval_topk=3):
    """
    This function evaluates a multi-centroid Nearest Class Mean (NCM) classifier within
    an associative memory model framework on a validation dataset.
     
    Specifically, it is designed to assess the model's performance in predicting the correct
    task associated with a set of classes, rather than predicting individual class labels directly.
    
    The evaluation includes computing the top-1 and top-k accuracies, along with the average number
    of retrievals required to correctly identify the task.
    """
    classes = torch.cat(task_split_classes, dim=0)
    train_features = train_features_dict['features']
    train_labels = train_features_dict['labels']
    train_dataset = FeatureDataset(train_features, train_labels)
    train_dataset.set_classes(classes)
    
    prototypes = []

    label_to_class_mapping = []

    for step, current_class in enumerate(classes):
        train_dataset.set_classes([current_class])
        current_features = train_dataset.data
        if cluster_method == 'spectral':
            clustering = spectral_clustering(current_features, n_clusters)
        elif cluster_method == 'Kmeans':
            clustering = Kmeans(current_features, n_clusters)
        elif cluster_method == 'BisectingKMeans':
            clustering = BisectingKmeans(current_features, n_clusters)
            
        for label in range(n_clusters):
            cluster_indices = clustering.labels_ == label
            cluster_features = current_features[cluster_indices]
            if not torch.is_tensor(cluster_features):
                cluster_features = torch.tensor(cluster_features)
            cluster_var, cluster_mean = torch.var_mean(cluster_features, dim=0)
            prototypes.append(cluster_mean)
            label_to_class_mapping.append(current_class)
    
    label_to_class_mapping = torch.tensor(label_to_class_mapping).long()
    
    prototypes = torch.stack(prototypes, dim=0)

    class_to_task_mapping = torch.zeros((len(classes),)).long()

    for task_id, task_classes in enumerate(task_split_classes):
        class_to_task_mapping[task_classes] = torch.tensor([task_id]).repeat(len(class_to_task_mapping[task_classes]))

    valid_features = valid_features_dict['features']
    valid_labels = valid_features_dict['labels']
    valid_task_labels = class_to_task_mapping[valid_labels]

    valid_dataset = FeatureDataset(valid_features, valid_labels)
    valid_dataset.set_classes(classes)

    features = torch.nn.functional.normalize(valid_features, dim=1)
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    similarity = torch.mm(features, prototypes.t())

    topk_values, topk_indices = similarity.topk(k=retrieval_topk)
    topk_predicted_labels = class_to_task_mapping[label_to_class_mapping[topk_indices]]

    top1_correct_predictions = torch.zeros_like(valid_task_labels).float()
    topk_correct_predictions = torch.zeros_like(valid_task_labels).float()
    retrieval_number = torch.zeros(len(valid_labels)).float()

    top1_correct_predictions = (topk_predicted_labels[:, 0] == valid_task_labels).float()

    for i in range(len(valid_task_labels)):
        if valid_task_labels[i] in topk_predicted_labels[i]:
            topk_correct_predictions[i] = 1.0
            retrieval_number[i] = len(torch.unique(topk_predicted_labels[i]))

    top1_average_accuracy = top1_correct_predictions.mean().item()
    topk_average_accuracy = topk_correct_predictions.mean().item()
    average_retrieval = retrieval_number.mean().item()

    return top1_average_accuracy, topk_average_accuracy, average_retrieval

# In[]
def spectral_clustering(features, n_clusters):
    affinity_matrix = utils.cosine_similarity(features, features)
    clustering = SpectralClustering(n_clusters=n_clusters, \
                assign_labels='discretize', affinity='precomputed', n_init=10)
    affinity_matrix = affinity_matrix.cpu().numpy()
    clustering.fit_predict(affinity_matrix)
    return clustering

def Kmeans(features, n_clusters):
    clustering = KMeans(n_clusters=n_clusters)
    clustering.fit(features.cpu().numpy())
    return clustering

def BisectingKmeans(features, n_clusters):
    clustering = BisectingKMeans(n_clusters=n_clusters)
    clustering.fit(features.cpu().numpy())
    return clustering

def execute_clustering(features, n_clusters, cluster_method):
    if cluster_method == 'spectral':
        clustering = spectral_clustering(features, n_clusters)
    elif cluster_method == 'Kmeans':
        clustering = Kmeans(features, n_clusters)
    elif cluster_method == 'BisectingKMeans':
        clustering = BisectingKmeans(features, n_clusters)
    return clustering
            
class MultiCentroidGaussianGenerator(Dataset):
    """
    The MultiCentroidGaussianGenerator class is designed to serve as a dataset generator
    for creating synthetic data points sampled from Gaussian distributions centered around
    learned centroids for each class. 
    """
    def __init__(self, classes, features_dict, n_clusters=3, cluster_method="spectral"):

        features = features_dict['features']
        labels = features_dict['labels']
        dataset = FeatureDataset(features, labels)

        self.label_to_class_mapping = torch.zeros((len(classes) * n_clusters,)).int()
        self.n_clusters = n_clusters
        self.labels = []
        self.dists = []

        for step, current_class in enumerate(classes):
            dataset.set_classes([current_class])
            current_features = dataset.data
            clustering = execute_clustering(current_features, n_clusters, cluster_method)
            
            for label in range(n_clusters):
                cluster_indexs = clustering.labels_ == label
                cluster_features = current_features[cluster_indexs]
                if not torch.is_tensor(cluster_features):
                    cluster_features = torch.tensor(cluster_features)
                cluster_var, cluster_mean = torch.var_mean(cluster_features, dim=0)

                self.dists.append(torch.distributions.MultivariateNormal(cluster_mean, torch.diag(cluster_var)))
                self.labels.append(step*n_clusters + label)
                self.label_to_class_mapping[step*n_clusters + label] = torch.tensor(current_class)

        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.dists)

    def __getitem__(self, idx):
        return self.dists[idx].sample(), self.labels[idx], self.label_to_class_mapping[idx]
    
class MultiCentroidGaussianGenerator_Task_Associative_Memory(Dataset):
    def __init__(self, task_split_classes, features_dict, n_clusters=3, cluster_method="spectral", diag=False):
        """
        This class is designed to serve as a custom PyTorch dataset generator for creating synthetic data points sampled
        from Gaussian distributions centered around learned centroids for each class.
        It is particularly suited for scenarios involving task-based associative memory,
        where different subsets of classes are grouped into tasks.
        """
        classes = torch.cat(task_split_classes, dim=0)
        features = features_dict['features']
        labels = features_dict['labels']
        dataset = FeatureDataset(features, labels)

        self.label_to_class_mapping = []
        self.n_clusters = n_clusters
        self.labels = []
        self.dists = []

        for step, current_class in enumerate(classes):
            dataset.set_classes([current_class])
            current_features = dataset.data
            clustering = execute_clustering(current_features, n_clusters, cluster_method)
            
            for label in range(n_clusters):
                cluster_indexs = clustering.labels_ == label
                cluster_features = current_features[cluster_indexs]
                if not torch.is_tensor(cluster_features):
                    cluster_features = torch.tensor(cluster_features)

                cluster_var, cluster_mean = torch.var_mean(cluster_features, dim=0)

                clustet_cov = torch.cov(cluster_features.t()) + 1e-3*torch.eye(cluster_features.size(1))

                if diag:
                    self.dists.append(torch.distributions.MultivariateNormal(cluster_mean, torch.diag(cluster_var)))
                else:
                    self.dists.append(torch.distributions.MultivariateNormal(cluster_mean, clustet_cov))

                self.labels.append(step*n_clusters + label)
                self.label_to_class_mapping.append(current_class)

        self.label_to_class_mapping = torch.tensor(self.label_to_class_mapping).long()
        
        self.labels = torch.tensor(self.labels).long()

        self.class_to_task_mapping = torch.zeros((len(classes),)).long()

        for task_id, task_classes in enumerate(task_split_classes):
            self.class_to_task_mapping[task_classes] = torch.tensor([task_id]).repeat(len(self.class_to_task_mapping[task_classes]))

    def __len__(self):
        return len(self.dists)

    def __getitem__(self, idx):
        return self.dists[idx].sample(), self.labels[idx], self.class_to_task_mapping[self.label_to_class_mapping[idx]]
    

class MultiCentroidGenerator_Task_Associative_Memory(Dataset):
    """Not a random dataset"""
    def __init__(self, task_split_classes, features_dict, n_clusters=3, cluster_method="spectral", diag=False):

        classes = torch.cat(task_split_classes, dim=0)
        features = features_dict['features']
        labels = features_dict['labels']
        dataset = FeatureDataset(features, labels)

        self.features = []
        self.label_to_class_mapping = []
        self.n_clusters = n_clusters
        self.labels = []

        for step, current_class in enumerate(classes):
            dataset.set_classes([current_class])
            current_features = dataset.data
            clustering = execute_clustering(current_features, n_clusters, cluster_method)
            
            for label in range(n_clusters):
                cluster_indexs = clustering.labels_ == label
                cluster_features = current_features[cluster_indexs]
                if not torch.is_tensor(cluster_features):
                    cluster_features = torch.tensor(cluster_features)

                self.features.append(cluster_features)

                self.labels.append(torch.tensor(step*n_clusters + label).long().repeat(len(cluster_features)))
                self.label_to_class_mapping.append(current_class)

        self.label_to_class_mapping = torch.tensor(self.label_to_class_mapping).long()
        
        self.labels = torch.cat(self.labels, dim=0)
        self.features = torch.cat(self.features, dim=0)

        self.class_to_task_mapping = torch.zeros((len(classes),)).long()

        for task_id, task_classes in enumerate(task_split_classes):
            self.class_to_task_mapping[task_classes] = torch.tensor([task_id]).repeat(len(self.class_to_task_mapping[task_classes]))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.class_to_task_mapping[
            self.label_to_class_mapping[self.labels[idx]]]
    
class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_features),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.net(x)

class MultiCentroidClassifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=512,
                 retrieval_topk=3, label_to_class_mapping=None, class_to_task_mapping=None, auxiliary_head=True):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features))
        self.dropout = nn.Dropout(p=0.0)

        self.ce = nn.CrossEntropyLoss()
        self.register_buffer('label_to_class_mapping', label_to_class_mapping)
    
        if class_to_task_mapping is not None:
            self.register_buffer('class_to_task_mapping', class_to_task_mapping)
            self.task_associate = True
            if auxiliary_head:
                task_num = class_to_task_mapping.max().int() + 1
                self.aux_head = nn.Linear(hidden_features, task_num)
                self.auxiliary_head = True
            else:
                self.auxiliary_head = False
        else:
            self.task_associate = False
            self.auxiliary_head = False

        self.retrieval_topk = retrieval_topk

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.dropout(x)
        # print(x.shape)
        if not self.auxiliary_head:
            return self.classifier(x), None
        else:
            return self.classifier(x), self.aux_head(x)

    def train_one_epoch(self, dataloader, optimizer, device):
        self.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, cluster, task_label in dataloader:
            inputs = inputs.to(device)
            cluster = cluster.to(device)
            task_label = task_label.to(device)
            optimizer.zero_grad()
            outputs, aux_outputs = self(inputs)
            if not self.auxiliary_head:
                loss = self.ce(outputs, cluster)
            else:
                # print(outputs.device)
                # print(cluster.device)
                # print(task_label.device)
                # print(aux_outputs.device)
                loss = self.ce(outputs, cluster) + 0.5 * self.ce(aux_outputs, task_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == cluster.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

    def train_model(self, train_loader, device, num_epochs):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        for epoch in range(num_epochs):
            print(f'Starting Epoch {epoch + 1}/{num_epochs}')
            self.train_one_epoch(train_loader, optimizer, device)
            scheduler.step()

    def evaluate_model(self, dataloader, device):
        self.to(device)
        self.eval()
        top1_correct_predictions = []
        topk_correct_predictions = []
        retrieval_number = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels
                outputs, aux_outputs = self(inputs)

                outputs = outputs.cpu()
                
                topk_values, topk_indices = outputs.topk(k=self.retrieval_topk, dim=1)

                topk_predicted_classes = self.label_to_class_mapping[topk_indices]

                if self.task_associate:
                    topk_predicted_classes = self.class_to_task_mapping[topk_predicted_classes]

                for i in range(len(labels)):
                    if labels[i] in topk_predicted_classes[i]:
                        topk_correct_predictions.append(1.0)
                    else:
                        topk_correct_predictions.append(0.0)
                    
                    top1_correct_predictions.append(1.0 if topk_predicted_classes[i][0] == labels[i] else 0.0)
                    retrieval_number.append(len(torch.unique(topk_predicted_classes[i])))

        top1_average_accuracy = sum(top1_correct_predictions) / len(top1_correct_predictions)
        topk_average_accuracy = sum(topk_correct_predictions) / len(topk_correct_predictions)
        average_retrieval_number = sum(retrieval_number) / len(retrieval_number)

        print(f'Valid Top-1 Retrieval Accuracy: {top1_average_accuracy:.4f}, '
              f'Valid Top-k Retrieval Accuracy: {topk_average_accuracy:.4f}, '
              f'Valid Average Retrieval Number: {average_retrieval_number:.4f}')
        
        return top1_average_accuracy, topk_average_accuracy, average_retrieval_number
    

# In[]
def evaluate_multi_centroid_classifier(
        classes, train_features_dict,
        valid_features_dict, n_clusters=3, cluster_method="spectral", retrieval_topk=3):
    
    train_dataset = MultiCentroidGaussianGenerator(
        classes, train_features_dict, n_clusters=n_clusters,
        cluster_method=cluster_method)
    
    valid_dataset = FeatureDataset(valid_features_dict['features'], valid_features_dict['labels'])

    valid_dataset.set_classes(classes)

    classifier = MLPClassifier(768, 100 * n_clusters)
    classifier = MultiCentroidClassifier(
        classifier,
        retrieval_topk=retrieval_topk,
        label_to_class_mapping=train_dataset.label_to_class_mapping)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.train_model(train_loader, device, num_epochs=60)

    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    top1_average_accuracy, topk_average_accuracy, average_retrieval_number = \
        classifier.evaluate_model(valid_loader, device)
    
    return top1_average_accuracy, topk_average_accuracy, average_retrieval_number

def evaluate_multi_centroid_classifier_task_associative(
        task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, cluster_method="spectral",
        retrieval_topk=3, diag=False, random=True, auxiliary_head=False):
    
    if random:
        train_dataset = MultiCentroidGaussianGenerator_Task_Associative_Memory(
            task_split_classes, train_features_dict, n_clusters=n_clusters,
            cluster_method=cluster_method, diag=diag)
        num_epochs = 60

    else:
        train_dataset = MultiCentroidGenerator_Task_Associative_Memory(
            task_split_classes, train_features_dict, n_clusters=n_clusters,
            cluster_method=cluster_method)
        
        num_epochs = 20
    
    valid_dataset = FeatureDataset(
        valid_features_dict['features'],
        valid_features_dict['labels']
        )
    
    label_to_class_mapping = train_dataset.label_to_class_mapping
    class_to_task_mapping = train_dataset.class_to_task_mapping
    
    valid_dataset.set_classes(torch.cat(task_split_classes, dim=0))

    valid_dataset.targets = class_to_task_mapping[valid_dataset.targets]

    classifier = MultiCentroidClassifier(
        768, 100 * n_clusters,
        retrieval_topk=retrieval_topk,
        label_to_class_mapping=label_to_class_mapping,
        class_to_task_mapping=class_to_task_mapping,
        auxiliary_head=auxiliary_head)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.train_model(train_loader, device, num_epochs=num_epochs)

    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    top1_average_accuracy, topk_average_accuracy, average_retrieval_number = \
        classifier.evaluate_model(valid_loader, device)
    
    return top1_average_accuracy, topk_average_accuracy, average_retrieval_number

def evaluate_multi_centroid_MLP_enhanced_NCM_task_associative(
        task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, cluster_method="spectral",
        retrieval_topk=3, diag=True, random=True, auxiliary_head=True):
    
    if random:
        train_dataset = MultiCentroidGaussianGenerator_Task_Associative_Memory(
            task_split_classes, train_features_dict, n_clusters=n_clusters,
            cluster_method=cluster_method, diag=diag)
        num_epochs = 60

    else:
        train_dataset = MultiCentroidGenerator_Task_Associative_Memory(
            task_split_classes, train_features_dict, n_clusters=n_clusters,
            cluster_method=cluster_method)
        
        num_epochs = 20
    
    label_to_class_mapping = train_dataset.label_to_class_mapping
    class_to_task_mapping = train_dataset.class_to_task_mapping
    
    classifier = MultiCentroidClassifier(
        768, 100 * n_clusters,
        hidden_features=1024,
        retrieval_topk=retrieval_topk,
        label_to_class_mapping=label_to_class_mapping,
        class_to_task_mapping=class_to_task_mapping,
        auxiliary_head=auxiliary_head)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.train_model(train_loader, device, num_epochs=num_epochs)
    
    train_dataset = FeatureDataset(train_features_dict['features'], train_features_dict['labels'])
    train_dataset.set_classes(torch.cat(task_split_classes, dim=0))
    valid_dataset = FeatureDataset(valid_features_dict['features'], valid_features_dict['labels'])
    valid_dataset.set_classes(torch.cat(task_split_classes, dim=0))

    with torch.no_grad():
        train_features = train_dataset.data
        train_features = classifier.feature_extractor.cpu()(train_features)
        valid_features = valid_dataset.data
        valid_features = classifier.feature_extractor.cpu()(valid_features)

    train_features_dict = copy.deepcopy(train_features_dict)
    valid_features_dict = copy.deepcopy(valid_features_dict)

    train_features_dict['features'] = train_features
    valid_features_dict['features'] = valid_features

    top1_average_accuracy, topk_average_accuracy, average_retrieval_number =\
          evaluate_multi_centroid_NCM_task_associative_memory(task_split_classes, train_features_dict, valid_features_dict, n_clusters=n_clusters,
            cluster_method=cluster_method, retrieval_topk=retrieval_topk)
    
    return top1_average_accuracy, topk_average_accuracy, average_retrieval_number

# In[]
class MCHN(object):
    def __init__(self, values, beta=2.0, similarity_function='dot_product'):
        super().__init__()
        self.values = values
        self.beta = beta
        self.similarity_function = similarity_function
        
    def Euclidean_distance(self, keys, queries):
        dist = torch.pow(keys.unsqueeze(1) - queries.unsqueeze(0), 2)
        return - dist.sum(dim=2)
    
    def dot_product_similarity(self, keys, queries):
        keys = torch.nn.functional.normalize(keys, dim=1)
        queries = torch.nn.functional.normalize(queries, dim=1)
        return keys @ queries.t()
    
    def Manhatten_distance(self, keys, queries):
        dist = torch.abs(keys.unsqueeze(1) - queries.unsqueeze(0))
        return - dist.sum(dim=2)
    
    def loop(self, queries, times=5):
        X = self.values.to(queries.device)
        for index in range(times):
            if self.similarity_function == 'dot_product':
                similarity = self.dot_product_similarity(X, queries)
            elif self.similarity_function == 'Manhatten':
                similarity = self.Manhatten_distance(X, queries)
            elif self.similarity_function == 'Euclidean':
                similarity = self.Euclidean_distance(X, queries)
            p = torch.softmax(self.beta * similarity, dim=0)
            queries = (X.t() @ p).t()
        return queries
    
#In[]
def random_projection(features_dict):
    random_projector = nn.Sequential(
        nn.Linear(768, 2048),
    )

    with torch.no_grad():
        features_dict['features'] = random_projector(features_dict['features'])
        features_dict['prototypes'] = random_projector(features_dict['prototypes'])

# In[]
def main(args):
    # set up misc
    utils.fix_random_seeds(args.seed)
    cudnn.deterministic = False
    cudnn.benchmark = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_tasks(args)

    train_features_dict, valid_features_dict = prepare_features(args)

    classes = torch.tensor(args.class_seq)

    evaluate_NCM_associative_memory(classes, train_features_dict, valid_features_dict)

    evaluate_multi_centroid_NCM_associative_memory(
        classes, train_features_dict,
        valid_features_dict,
        cluster_method=args.cluster_method, n_clusters=args.n_clusters)
    
    evaluate_multi_centroid_classifier(
        classes, train_features_dict,
        valid_features_dict,
        cluster_method=args.cluster_method, n_clusters=args.n_clusters)
    
    task_split_classes = torch.tensor(args.class_seq)[torch.randperm(100)].reshape(10, -1)
    task_split_classes = [task_split_classes[i] for i in range(len(task_split_classes))]

    # random_projection(train_features_dict)
    # random_projection(valid_features_dict)

    evaluate_multi_centroid_NCM_task_associative_memory(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, cluster_method='Kmeans')
    
    evaluate_multi_centroid_NCM_task_associative_memory(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=5, retrieval_topk=5, cluster_method='spectral')

    evaluate_multi_centroid_NCM_task_associative_memory(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, cluster_method='BisectingKMeans')

    evaluate_multi_centroid_classifier_task_associative(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, diag=True, auxiliary_head=False)
    
    evaluate_multi_centroid_classifier_task_associative(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, diag=True, random=False, auxiliary_head=True)
    
    evaluate_multi_centroid_classifier_task_associative(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, diag=True, random=True, auxiliary_head=True)
    
    evaluate_multi_centroid_MLP_enhanced_NCM_task_associative(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, diag=True, random=True, auxiliary_head=True)
    
    evaluate_multi_centroid_classifier_task_associative(task_split_classes, train_features_dict,
        valid_features_dict, n_clusters=3, retrieval_topk=6, diag=False)
    

# In[]
if __name__ == "__main__":
    s_t = time.time()
    parser = get_args_parser()
    args = parser.parse_args(args=[])
    torch.set_printoptions(precision=4)
    main(args)
    e_t = time.time()
    print('Time Usage:{}'.format(np.around((e_t-s_t)/3600, 2)))