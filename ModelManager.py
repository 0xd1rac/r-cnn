import torchvision.models as models
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import os 

class ModelManager:
    @staticmethod
    def get_region_proposal_classifier(label_map):
        model = models.vgg16(pretrained=True)
        num_classes_coco = len(label_map.keys())
        model.classifier[6] = torch.nn.Linear(in_features=4096,
                                      out_features = num_classes_coco,
                                      bias=True
                                     )
        return model
    
    @staticmethod
    def train_region_proposal_classifier(model, 
                                         num_epochs,
                                         train_loader, 
                                         device,
                                         model_folder
                                         ):

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model.train()
        model.to(device)

        for epoch in range(num_epochs):
            running_loss = 0.0

            for inputs, labels in train_loader:
                # Move inputs and labels to GPU if available
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

            # Save the model checkpoint after each epoch
            checkpoint_path = os.path.join(model_folder, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }, checkpoint_path)

            print(f'Model checkpoint saved: {checkpoint_path}')

        print('Training complete')

    @staticmethod
    def extract_features(feature_extractor, loader, device):
        features = []
        labels = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                outputs = feature_extractor(inputs)
                features.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())  # Convert targets to numpy arrays

        # Convert lists of arrays to single numpy arrays
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        return features, labels

    @staticmethod
    def train_svm(train_features, train_labels):
        svm = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        svm.fit(train_features, train_labels)

        return svm