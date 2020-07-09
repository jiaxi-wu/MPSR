from PIL import Image
from maskrcnn_benchmark.data.datasets.voc import PascalVOCDataset
import os, shutil
import sys
#crop the object from original image and save them in original shape
#save object under categorized folders
split = int(sys.argv[1])


#here we do not crop the original size, but crop a (8 / 7) larger closeup
def get_closeup(image, target):
    closeup = []
    closeup_target = target.get_field('labels').tolist()
    for t in range(len(target)):
        x1, y1, x2, y2 = target.bbox[t].tolist()
        cutsize = max(x2 - x1, y2 - y1) * 8 / 7 / 2 
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        crop_img = image.crop((int(midx - cutsize), int(midy - cutsize), int(midx + cutsize), int(midy + cutsize)))
        closeup.append(crop_img)
    return closeup, closeup_target


datadirs = ['datasets/voc/VOC2007', 'datasets/voc/VOC2012']
splits = ['trainval_split%d_base'%split, 'trainval_split%d_base'%split]
for s in range(2):
    dataset = PascalVOCDataset(datadirs[s], splits[s])
    if not os.path.exists(datadirs[s] + '/Crops'):
        os.mkdir(datadirs[s] + '/Crops')
    else:
        shutil.rmtree(datadirs[s] + '/Crops')
        os.mkdir(datadirs[s] + '/Crops')
    for cls in PascalVOCDataset.CLASSES[1:]:
        os.mkdir(datadirs[s] + '/Crops/' + cls)
    cls_count = {cls: 0 for cls in PascalVOCDataset.CLASSES}
    for index in range(len(dataset)):
        img_id = dataset.ids[index]
        img = Image.open(datadirs[s] + '/JPEGImages/%s.jpg'%img_id).convert("RGB")
        annos = dataset.get_groundtruth(index)
        crops, crop_labels = get_closeup(img, annos)
        for crop, label in list(zip(crops, crop_labels)):
            #label = PascalVOCDataset.CLASSES[label]
            label = dataset.categories[label]
            cls_count[label] += 1
            crop.save(datadirs[s] + '/Crops/%s/%d.jpg'%(label, cls_count[label]))
    print(cls_count)
    print('crop amount:%d'%sum(list(cls_count.values())))
