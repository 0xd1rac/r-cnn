import torch
from torch.utils.data import Dataset, DataLoader
import selectivesearch
import numpy as np
from PIL import Image
import random

class CustomObjectDetectionDataset(Dataset):
    def __init__(self, dataset, label_map, crop_size=(224, 224), transform=None):
        """
        Args:
            dataset: A dataset of (image, target) tuples.
            label_map: A dictionary mapping class labels to integers.
            crop_size: Tuple specifying the (width, height) to which all crops will be resized.
            transform: Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.label_map = label_map
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)
        inter_area = inter_width * inter_height

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def perform_selective_search(self, image):
        img_np = np.array(image)
        _, regions = selectivesearch.selective_search(img_np, scale=500, sigma=0.9, min_size=10)

        proposals = []
        for r in regions:
            if r['size'] >= 200:
                x, y, w, h = r['rect']
                proposals.append([x, y, x + w, y + h])

        return proposals

    def label_proposals_with_ground_truth(self, proposals, gt_boxes, gt_labels):
        background_proposals, object_proposals = [], []
        for proposal in proposals:
            max_iou, assigned_label = 0, self.label_map['background']

            for i, gt_box in enumerate(gt_boxes):
                iou = self.calculate_iou(proposal, gt_box)
                if iou >= 0.5 and iou > max_iou:
                    max_iou, assigned_label = iou, gt_labels[i]

            proposal_tuple = (proposal, assigned_label)
            if assigned_label == self.label_map['background']:
                background_proposals.append(proposal_tuple)
            else:
                object_proposals.append(proposal_tuple)

        return background_proposals, object_proposals

    def get_crops(self, image, proposals):
        crops = []
        for proposal, _ in proposals:
            x_min, y_min, x_max, y_max = proposal
            crop = image.crop((x_min, y_min, x_max, y_max)).resize(self.crop_size, Image.Resampling.LANCZOS)
            crop = self.transform(crop) if self.transform else torch.tensor(np.array(crop)).permute(2, 0, 1).float() / 255.0
            crops.append(crop)

        return torch.stack(crops) if crops else torch.tensor([])

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        gt_boxes, gt_labels = target['boxes'].numpy(), target['labels'].numpy()

        proposals = self.perform_selective_search(image)
        labeled_background_proposals, labeled_object_proposals = self.label_proposals_with_ground_truth(proposals, gt_boxes, gt_labels)

        image_tensor = self.transform(image) if self.transform else torch.tensor(np.array(image.resize(self.crop_size, Image.Resampling.LANCZOS))).permute(2, 0, 1).float() / 255.0

        background_proposal_crops = self.get_crops(image, labeled_background_proposals)
        object_proposal_crops = self.get_crops(image, labeled_object_proposals)
        gt_crops = self.get_crops(image, list(zip(gt_boxes, gt_labels)))

        sample = {
            'image': image_tensor,
            'gt_boxes': torch.tensor(gt_boxes, dtype=torch.float32),
            'gt_labels': torch.tensor(gt_labels, dtype=torch.int64),
            'gt_crops': gt_crops,
            'background_proposal_crops': background_proposal_crops,
            'object_proposal_crops': object_proposal_crops,
            'object_proposal_labels': torch.tensor([p[1] for p in labeled_object_proposals], dtype=torch.int64)
        }

        return sample


def sample_proposals_and_labels(train_custom_dataset, max_samples=5):
    train_imgs, train_labels = [], []

    for sample in train_custom_dataset:
        object_samples, object_labels = sample_proposals(sample['object_proposal_crops'], sample['object_proposal_labels'], max_samples)
        background_samples = sample_proposals(sample['background_proposal_crops'], torch.zeros(len(sample['background_proposal_crops'])), max_samples)

        train_imgs.extend(object_samples + background_samples)
        train_labels.extend(object_labels + [0] * len(background_samples))

    train_imgs = torch.stack(train_imgs)
    train_labels = torch.tensor(train_labels, dtype=torch.int64)

    return train_imgs, train_labels


def sample_proposals(proposal_crops, proposal_labels, max_samples):
    num_proposals = len(proposal_crops)
    if num_proposals > max_samples:
        sampled_indices = random.sample(range(num_proposals), max_samples)
        samples = [proposal_crops[i] for i in sampled_indices]
        labels = [proposal_labels[i] for i in sampled_indices]
    else:
        samples, labels = proposal_crops, proposal_labels

    return samples, labels
