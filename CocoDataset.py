import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class CocoDataset(Dataset):
    def __init__(self, fiftyone_dataset, label_map, transform=None):
        self.dataset = fiftyone_dataset
        self.label_map = label_map
        self.transform = transform
        self.filepaths = self.dataset.values("filepath")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        sample = self.dataset.match({"filepath": filepath}).first()
        image = Image.open(filepath)

        # Check if ground_truth exists
        if sample.ground_truth is None or sample.ground_truth.detections is None:
            # If no ground truth, return an empty target
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Get the bounding boxes and labels
            boxes = []
            labels = []
            for detection in sample.ground_truth.detections:
                x, y, w, h = detection.bounding_box
                # Convert bounding box to absolute coordinates
                x_min = x * image.width
                y_min = y * image.height
                x_max = (x + w) * image.width
                y_max = (y + h) * image.height
                boxes.append([x_min, y_min, x_max, y_max])

                # Convert the label from string to integer
                label = self.label_map.get(detection.label, 0)  # default to 0 if label not found
                labels.append(label)

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}
        return image, target
