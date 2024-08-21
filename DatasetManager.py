from CocoDataset import CocoDataset
import fiftyone as fo
import fiftyone.zoo as foz
from CustomObjectDetectionDataset import CustomObjectDetectionDataset, sample_proposals_and_labels
from torch.utils.data import TensorDataset, DataLoader
from RegionProposalDataset import RegionProposalDataset


class DatasetManager:
    @staticmethod
    def get_coco_datasets(train_transform=None,
                      val_transform=None,
                      test_transform=None,
                      max_train_samples=3,
                      max_val_samples=2,
                      max_test_samples=2
                      ):
        train_dataset = foz.load_zoo_dataset("coco-2017", split="train", max_samples=max_train_samples)
        val_dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=max_val_samples)
        test_dataset = foz.load_zoo_dataset("coco-2017", split="test", max_samples=max_test_samples)

        unique_labels = train_dataset.distinct("ground_truth.detections.label")
        label_map = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        label_map['background'] = 0

        train_coco_dataset = CocoDataset(train_dataset, label_map, transform=train_transform)
        val_coco_dataset = CocoDataset(val_dataset, label_map, transform=val_transform)
        test_coco_dataset = CocoDataset(test_dataset, label_map, transform=test_transform)

        return train_coco_dataset, val_coco_dataset, test_coco_dataset, label_map

    @staticmethod
    def get_dl_for_region_proposal_classifier():
        train_coco_dataset, val_coco_dataset, test_coco_dataset, label_map = DatasetManager.get_coco_datasets()
        train_custom_dataset = CustomObjectDetectionDataset(train_coco_dataset, label_map, transform=None)
        val_custom_dataset = CustomObjectDetectionDataset(val_coco_dataset, label_map, transform=None)
        test_custom_dataset = CustomObjectDetectionDataset(test_coco_dataset, label_map, transform=None)
        
        train_proposal_dataset = RegionProposalDataset(train_custom_dataset, label_map)
        val_proposal_dataset = RegionProposalDataset(val_custom_dataset, label_map)
        test_proposal_dataset = RegionProposalDataset(test_custom_dataset, label_map)

        train_loader = DataLoader(train_proposal_dataset, batch_size=5, shuffle=True)
        val_loader = DataLoader(val_proposal_dataset, batch_size=5, shuffle=True)
        test_loader = DataLoader(test_proposal_dataset, batch_size=5, shuffle=True)

        return train_loader, val_loader, test_loader, label_map