import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import ViTFeatureExtractor


class ImageClassificationCollator:
   def __init__(self, feature_extractor): 
      self.feature_extractor = feature_extractor
   def __call__(self, batch):  
      encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')   
      encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
      return encodings

def get_dataloader(dataset_folder, BATCH_SIZE, val_split=0.20, pretrained_model_path="google/vit-base-patch16-224-in21k"):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(dataset_folder, transform=transform)
    num_class = len(dataset.classes)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=42)
    datasets = {}
    train_idx, val_idx = train_test_split(list(range(len(Subset(dataset, train_idx)))), test_size=val_split, random_state=42)
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    
    print(f"Train dataset : {len(datasets['train'])}")
    print(f"Validation dataset : {len(datasets['val'])}")
    print(f"Test dataset : {len(datasets['test'])}")
    
    feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_path)
    collator = ImageClassificationCollator(feature_extractor)
    train_loader = DataLoader(datasets['train'], batch_size=8, collate_fn=collator, num_workers=16, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=8, collate_fn=collator, num_workers=16)
    test_loader = DataLoader(datasets['test'], batch_size=8, collate_fn=collator, num_workers=16)

    return train_loader, val_loader, test_loader, num_class