from PIL import Image
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.data.datasets.closeup import CloseupDataset
import os
#crop the object from original image and save them in original shape
#save object under categorized folders


#here we do not crop the original size, but crop a (8 / 7) larger closeup
def get_closeup(image, target):
    closeup = []
    closeup_target = target.get_field('labels').tolist()
    for t in range(len(target)):
        x1, y1, x2, y2 = target.bbox[t].tolist()
        if min(x2 - x1, y2 - y1) < 8:
            continue
        cutsize = max(x2 - x1, y2 - y1) * 8 / 7 / 2 
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        crop_img = image.crop((int(midx - cutsize), int(midy - cutsize), int(midx + cutsize), int(midy + cutsize)))
        closeup.append(crop_img)
    return closeup, closeup_target


imgdirs = ['datasets/coco/train2014', 'datasets/coco/val2014']
annofiles = ["datasets/coco/annotations/instances_train2014_base.json", "datasets/coco/annotations/instances_valminusminival2014_base.json"]
if not os.path.exists('datasets/coco/Crops'):
    os.mkdir('datasets/coco/Crops')
else:
    shutil.rmtree('datasets/coco/Crops')
    os.mkdir('datasets/coco/Crops')
for cls in CloseupDataset.CLASSES_COCO_BASE:
    os.mkdir('datasets/coco/Crops/' + cls)
cls_count = {cls: 0 for cls in CloseupDataset.CLASSES_COCO_BASE}
for s in range(2):
    dataset = COCODataset(annofiles[s], imgdirs[s], True)
    for index in range(len(dataset)):
        img, annos, _ = dataset.__getitem__(index)
        crops, crop_labels = get_closeup(img, annos)
        for crop, label in list(zip(crops, crop_labels)):
            label = CloseupDataset.CLASSES_COCO_BASE[label]
            cls_count[label] += 1
            crop.save('datasets/coco/Crops/%s/%d.jpg'%(label, cls_count[label]))
    print(cls_count)
    print('crop amount:%d'%sum(list(cls_count.values())))
