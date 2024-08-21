import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PlotManager:
    @staticmethod
    def visualize_random_sample_from_coco(dataset, label_map):
        # Select a random sample (image, target) tuple from the dataset
        image, target = random.choice(dataset)

        # Convert the PIL image to a numpy array
        image = np.array(image)

        # Plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Get the bounding boxes and labels from the target
        boxes = target['boxes'].numpy()  # Assuming boxes are already in pixel coordinates
        labels = target['labels'].numpy()

        # Reverse the label map to get the label names
        reverse_label_map = {v: k for k, v in label_map.items()}

        # Plot the bounding boxes and labels
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add the label text
            label = reverse_label_map.get(labels[i], "unknown")
            plt.text(x_min, y_min, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        # Display the plot with the image, bounding boxes, and labels
        plt.axis('off')  # Hide axes
        plt.show()