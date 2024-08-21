from DatasetManager import DatasetManager
from PlotManager import PlotManager
from CustomObjectDetectionDataset import CustomObjectDetectionDataset
from RegionProposalDataset import RegionProposalDataset
import fiftyone.zoo as foz
from CocoDataset import CocoDataset

from ModelManager import ModelManager
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


train_dl, val_dl, test_dl, label_map = DatasetManager.get_dl_for_region_proposal_classifier()
model = ModelManager.get_region_proposal_classifier(label_map)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_folder_path = "models/region_proposal_classifier"
ModelManager.train_region_proposal_classifier(model, 
                                              num_epochs=3, 
                                              train_loader=train_dl, 
                                              device=device,
                                              model_folder=model_folder_path
                                              )


# model = ModelManager.get_region_proposal_classifier(label_map)
# ModelManager.train_region_proposal_classifier(model, num_epochs=3, train_loader=train_dl, device=device)


# feature_extractor = model
# feature_extractor.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# feature_extractor.eval()
# feature_extractor.to(device)

# train_features, train_labels = ModelManager.extract_features(feature_extractor, train_dl, device)
# svm = ModelManager.train_svm(train_features, train_labels)

# val_features, val_labels = ModelManager.extract_features(feature_extractor, val_dl, device)
# val_predictions = svm.predict(val_features)
# accuracy = accuracy_score(val_labels, val_predictions)
# print(f"Validation Accuracy: {accuracy:.4f}")
