from maskrcnn_benchmark.data.datasets.coco import COCODataset
import json
import random
random.seed(0)
yolodir = '../Fewshot_Detection'
train = json.load(open('datasets/coco/annotations/instances_train2014.json'))
val = json.load(open('datasets/coco/annotations/instances_valminusminival2014.json'))
minival = json.load(open('datasets/coco/annotations/instances_minival2014.json'))
minival_ids = [i['id'] for i in minival['images']]
cls_dict = {v['id']: k for k, v in enumerate(train['categories'])}


images = {}
annotations = {}

for data in [train, val]:
    for im in data['images']:
        images[im['id']] = im
        annotations[im['id']] = []
for data in [train, val]:
    for anno in data['annotations']:
        annotations[anno['image_id']].append(anno)

with open(yolodir + '/data/coco.names') as f:
    cls = f.readlines()
    cls = [i.strip() for i in cls]

for shot in [10, 30]:
    ids_total = []
    annos_total = []
    for split in ['train', 'val']:
        ids = []
        for c in cls:
            with open(yolodir + '/data/cocosplit/full_box_%dshot_%s_trainval.txt'%(shot, c)) as f:
                content = f.readlines()
                ids += [int(i.strip()[-16: -4]) for i in content if split in i]
        ids = list(set(ids))
        ids = [i for i in ids if i not in minival_ids]
        ids_total += ids
        annos = []
        for i in ids:
            annos += annotations[i]
        annos_total += annos
        #json.dump(n_json, open('./datasets/coco/annotations/instances_%s2014_%dshot_novel_standard.json'%(split, shot), 'w+'))
    
    
    annos_total = [anno for anno in annos_total if anno['iscrowd'] == 0]
    cls_count = [0] * 80
    for anno in annos_total:
        cls_count[cls_dict[anno['category_id']]] += 1
    print(len(ids_total))
    print(cls_count)
    img_ids = list(images.keys())
    random.shuffle(img_ids)
    for img_id in img_ids:
        append_cls = [cls_dict[anno['category_id']] for anno in annotations[img_id] if anno['iscrowd']==0]
        if len(append_cls) == 0:
            continue
        tmp_count = [i for i in cls_count]
        for i in append_cls:
            tmp_count[i] += 1
        if max(tmp_count) > shot:
            continue
        if img_id not in ids_total:
            ids_total.append(img_id)
            cls_count = tmp_count
            if min(cls_count) == 30:
                break
    print(len(ids_total))
    print(cls_count)
    ids_total = sorted(list(set(ids_total)))
    for split in ['train', 'val']:
        n_json = {'info': train['info'],
                  'images': [],
                  'licenses': train['licenses'],
                  'annotations': [],
                  'categories': train['categories']}
        for i in ids_total:
            if split in images[i]['file_name']:
                n_json['images'].append(images[i])
                n_json['annotations'] += annotations[i]
        json.dump(n_json, open('./datasets/coco/annotations/instances_%s2014_%dshot_novel_standard.json'%(split, shot), 'w+'))

