import torch
checkpoint = torch.load('model_coco_base.pth', map_location=torch.device("cpu"))
model = checkpoint['model']
aimclass = 81
change = [('module.roi_heads.box.predictor.cls_score.weight', (aimclass, 1024)), 
          ('module.roi_heads.box.predictor.cls_score.bias'  , aimclass)]
t = torch.empty(change[0][1])
torch.nn.init.normal_(t, std=0.001)
model[change[0][0]] = t
t = torch.empty(change[1][1])
torch.nn.init.constant_(t, 0)
model[change[1][0]] = t
checkpoint = dict(model=model)
torch.save(checkpoint, 'coco_base_pretrained.pth')
