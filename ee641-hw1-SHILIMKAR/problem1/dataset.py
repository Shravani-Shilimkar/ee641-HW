import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pycocotools.coco import COCO
import torchvision.transforms.v2 as T

class ShapeDetectionDataset(Dataset):
    """
    A PyTorch Dataset for loading the synthetic shape detection data.
    """
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotation_file (str): Path to the COCO-style JSON annotation file.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        if transforms is None:
            self.transforms = T.Compose([
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        """
        Retrieves an image and its annotations.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_path = os.path.join(self.root_dir, coco.loadImgs(img_id)[0]['file_name'])
        image = read_image(img_path)
        
        # Prepare target annotations
        boxes = []
        labels = []
        for ann in anns:
            # COCO format is [x, y, width, height], convert to [x1, y1, x2, y2]
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)