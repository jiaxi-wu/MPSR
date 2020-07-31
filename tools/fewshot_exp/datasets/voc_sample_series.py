from maskrcnn_benchmark.data.datasets.voc import PascalVOCDataset
import random
from collections import OrderedDict
import numpy as np
import sys
# splits: 1. 5classes together, 2. 1 of 5 classes only
# Warning: only few-shot classes are limited to N-shot here (not base classes).
# We do this for more general sampling.

def get_groundtruth(voc07, voc12, ind):
    if ind < len(voc07):
        return voc07.get_groundtruth(ind)
    else:
        return voc12.get_groundtruth(ind - len(voc07))

#1.random sample from 
shot = int(sys.argv[1])#10
split = int(sys.argv[2])#1
random.seed(int(sys.argv[3]))#0
dataset07 = PascalVOCDataset('datasets/voc/VOC2007', 'trainval')
dataset12 = PascalVOCDataset('datasets/voc/VOC2012', 'trainval')
dataset_len = len(dataset07) + len(dataset12)
all_cls = PascalVOCDataset.CLASSES
novel_cls = [PascalVOCDataset.CLASSES_SPLIT1_NOVEL, 
             PascalVOCDataset.CLASSES_SPLIT2_NOVEL,
             PascalVOCDataset.CLASSES_SPLIT3_NOVEL,][split - 1]
novel_index = [all_cls.index(c) for c in novel_cls]

for series in range(1, 2):
    img_count = []
    exclude = [[] for i in range(len(all_cls))]
    include = []
    for i in range(dataset_len):
        anno = get_groundtruth(dataset07, dataset12, i)
        label = anno.get_field('labels')
        count = [(label == j).sum().item() for j in novel_index]
        if sum(count) == 0:
            for j in list(set(label.tolist())):
                exclude[j].append(i)
        else:
            include.append(i)
    
    random.shuffle(include)
    for i in range(len(exclude)):
        random.shuffle(exclude[i])

    pick = []
    box_count = np.zeros((len(all_cls)))

    for i in include:
        anno = get_groundtruth(dataset07, dataset12, i)
        label = anno.get_field('labels').tolist()
        tmp_count = box_count.copy()
        for j in label:
            tmp_count[j] += 1
        if tmp_count[novel_index].max() > shot:
            continue
        else:
            pick.append(i)
            for j in label:
                box_count[j] += 1
    for cls_ind, base_single in enumerate(exclude):
        if len(base_single) > 0:
            for i in base_single:
                if box_count[cls_ind] > shot:
                    break
                pick.append(i)
                anno = get_groundtruth(dataset07, dataset12, i)
                label = anno.get_field('labels').tolist()
                for j in label:
                    box_count[j] += 1

    box_count = np.zeros((len(all_cls)))
    for i in pick:
        anno = get_groundtruth(dataset07, dataset12, i)
        label = anno.get_field('labels').tolist()
        for j in label:
            box_count[j] += 1
    pick = sorted(list(set(pick)))
    print("___series %d : %d___"%(series, len(pick)))
    print(dict(zip(all_cls, box_count.tolist())))
    with open('./datasets/voc/VOC2007/ImageSets/Main/trainval_%dshot_novel_standard.txt'%(shot), 'w+') as f:
        for i in pick:
            if i < len(dataset07):
                f.write(dataset07.ids[i] + '\n')
    with open('./datasets/voc/VOC2012/ImageSets/Main/trainval_%dshot_novel_standard.txt'%(shot), 'w+') as f:
        for i in pick:
            if i >= len(dataset07):
                f.write(dataset12.ids[i - len(dataset07)] + '\n')
