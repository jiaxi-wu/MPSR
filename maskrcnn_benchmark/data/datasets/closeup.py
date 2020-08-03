import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class CloseupDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT_1 = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT_2 = (
        "__background__ ",
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT_3 = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    )

    CLASSES_COCO_NOVEL = (
        'airplane', 'bicycle', 'bird', 'boat', 
        'bottle', 'bus', 'car', 'cat', 'chair', 
        'cow', 'dining table', 'dog', 'horse', 
        'motorcycle', 'person', 'potted plant', 
        'sheep', 'couch', 'train', 'tv'
    )

    CLASSES_COCO_BASE = (
        '__background__', 'truck', 'traffic light', 'fire hydrant', 
        'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
        'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'wine glass', 
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
        'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' 
    )

    CLASSES_COCO = (
       '__background__', 'person', 'bicycle', 'car', 'motorcycle', 
       'airplane', 'bus', 'train', 'truck', 'boat', 
       'traffic light', 'fire hydrant', 'stop sign', 
       'parking meter', 'bench', 'bird', 'cat', 'dog', 
       'horse', 'sheep', 'cow', 'elephant', 'bear', 
       'zebra', 'giraffe', 'backpack', 'umbrella', 
       'handbag', 'tie', 'suitcase', 'frisbee', 
       'skis', 'snowboard', 'sports ball', 'kite', 
       'baseball bat', 'baseball glove', 'skateboard', 
       'surfboard', 'tennis racket', 'bottle', 'wine glass', 
       'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
       'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
       'book', 'clock', 'vase', 'scissors', 'teddy bear', 
       'hair drier', 'toothbrush'         
    )


    def __init__(self, data_dir, split, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.transforms = transforms

        path = "Crops_standard" if 'standard' in split else "Crops"
        if 'extreme' in split:
            path = "Crops_extreme"
        self._imgpath = os.path.join(self.root, path, "%s")

        self.ids = []
        self.labels = []

        if 'split1_base' in split:
            cls = CloseupDataset.CLASSES_SPLIT_1
        elif 'split2_base' in split:
            cls = CloseupDataset.CLASSES_SPLIT_2
        elif 'split3_base' in split:
            cls = CloseupDataset.CLASSES_SPLIT_3
        elif 'coco_base' in split:
            cls = CloseupDataset.CLASSES_COCO_BASE
        elif 'coco_standard' in split:
            cls = CloseupDataset.CLASSES_COCO
        else:
            cls = CloseupDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        
        count = {c:0 for c in cls[1:]}
        for i, c in enumerate(cls[1:]):
            class_dir = os.path.join(self.root, path, c)
            for img_id in os.listdir(class_dir):
                self.ids.append(c + "/" + img_id)
                self.labels.append(i + 1)
                count[c] += 1
        print(count)
        
        #too few ids lead to an unfixed bug
        if len(self.ids) < 50:
            self.ids = self.ids * (int(100 / len(self.ids)) + 1)
            self.labels = self.labels * (int(100 / len(self.labels)) + 1)
        
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}




    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")
        target = self.labels[index]
        
        imgs = []
        if self.transforms is not None:
            for t in self.transforms:
                imgs.append(t(img, None))
        target = torch.Tensor([target], device=imgs[0].device)
        return imgs, target

    def __len__(self):
        return len(self.ids)

    '''
    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}
    '''
