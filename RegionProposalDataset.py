from torch.utils.data import Dataset, DataLoader
import torch
import selectivesearch
import numpy as np
from PIL import Image
import random
from CustomObjectDetectionDataset import CustomObjectDetectionDataset

class RegionProposalDataset(Dataset):
    def __init__(self, 
                 custom_dataset: CustomObjectDetectionDataset, 
                 label_map, 
                 max_object_proposal_sample=5,
                 max_background_proposal_sample=5,
                 crop_size=(224, 224), 
                 transform=None
                 ):
    
        self.custom_dataset = custom_dataset
        self.label_map = label_map
        self.crop_size = crop_size
        self.transform = transform
        self.max_object_proposal_sample = max_object_proposal_sample
        self.max_background_proposal_sample = max_background_proposal_sample
        self.sample_proposals_and_labels()
    
    def __len__(self):
        return len(self.imgs)

    def sample(self, proposal_crops, proposal_labels, max_samples):
        num_proposals = len(proposal_crops)
        if num_proposals > max_samples:
            sampled_indices = random.sample(range(num_proposals), max_samples)
            samples = [proposal_crops[i] for i in sampled_indices]
            labels = [proposal_labels[i] for i in sampled_indices]
        else:
            samples, labels = proposal_crops, proposal_labels
        
        return samples, labels

    def sample_proposals_and_labels(self):
        imgs, labels = [], []
        for sample in self.custom_dataset:
            object_samples, object_labels = self.sample(sample['object_proposal_crops'], 
                                                        sample['object_proposal_labels'], 
                                                        self.max_object_proposal_sample
                                                        )
            background_samples, _ = self.sample(sample['background_proposal_crops'], 
                                                                torch.zeros(len(sample['background_proposal_crops'])), 
                                                                self.max_background_proposal_sample
                                                                )
            imgs.extend(object_samples)
            imgs.extend(background_samples)
            labels.extend(object_labels)
            labels.extend([0] * len(background_samples))
    
        imgs_tensor = torch.stack(imgs)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        self.imgs = imgs_tensor
        self.labels = labels_tensor
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        return img, label
       
