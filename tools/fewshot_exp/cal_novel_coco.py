from maskrcnn_benchmark.data.datasets.closeup import CloseupDataset
import sys
all_cls = CloseupDataset.CLASSES_COCO[1:]
#novel_cls = [c for c in CloseupDataset.CLASSES_COCO if c not in CloseupDataset.CLASSES_COCO_BASE]
novel_cls = CloseupDataset.CLASSES_COCO_NOVEL
novel_idx = [all_cls.index(i) for i in novel_cls]
for shot in [10, 30]:
    novel_mmap = [0] * 20
    novel_map = [0] * 20
    try:
        with open(sys.argv[1] + '/result_%dshot.txt'%shot) as f:
            results = f.readlines()
        #mmap
        mmaps = results[2: 82]
        mmaps = [float(i.strip()) for i in mmaps]
        for i, j in enumerate(novel_idx):
            novel_mmap[i] += mmaps[j]
        #map
        maps = results[83: 163]
        maps = [float(i.strip()) for i in maps]
        for i, j in enumerate(novel_idx):
            novel_map[i] += maps[j]
        print('result of %d shot:'%shot)
        print('novel mmap:%.4f'%(sum(novel_mmap) / 20))
        print(dict(zip(novel_cls, novel_mmap)))
        print('novel map:%.4f'%(sum(novel_map) / 20))
        print(dict(zip(novel_cls, novel_map)))
        print('')
    except Exception as e:
        print('result file: %d shot not found'%shot)
        continue
