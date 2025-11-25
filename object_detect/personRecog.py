# ============================================================
# 🚶‍♂️ FASTER R-CNN FOR PEDESTRIAN DETECTION (COCO PERSON CLASS)
# ============================================================

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.models.detection as detection
import torchvision.transforms.functional as F
import torchvision.ops as ops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# ======================================
# 1️⃣ Detection Transform (Fix for PIL/Tensor conflict)
# ======================================
class DetectionTransform:
    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, target):
        # Convert to tensor only if not already
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)

        # Random horizontal flip for training
        if self.train and random.random() < 0.5:
            width = image.shape[-1]
            image = F.hflip(image)
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


# ======================================
# 2️⃣ Dataset for COCO Pedestrian Detection
# ======================================
class CocoPedestrian(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)

        # Keep only 'person' category (category_id == 1)
        anno = [obj for obj in anno if obj["category_id"] == 1]

        boxes, labels, areas, iscrowd = [], [], [], []
        for obj in anno:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            areas.append(w * h)
            iscrowd.append(obj.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.uint8),
        }

        # Apply transforms safely
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


# ======================================
# 3️⃣ Utility Functions
# ======================================
def get_transform(train):
    return DetectionTransform(train=train)

def collate_fn(batch):
    return tuple(zip(*batch))


# ======================================
# 4️⃣ Initialize Datasets & Loaders
# ======================================
train_root = "/Users/pubalimazumder/datasets/coco/train2017"
train_ann = "/Users/pubalimazumder/datasets/coco/annotations/instances_train2017.json"
val_root = "/Users/pubalimazumder/datasets/coco/val2017"
val_ann = "/Users/pubalimazumder/datasets/coco/annotations/instances_val2017.json"

train_ds = CocoPedestrian(train_root, train_ann, transforms=get_transform(train=True))
val_ds   = CocoPedestrian(val_root, val_ann, transforms=get_transform(train=False))

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)


# ======================================
# 5️⃣ Model: Faster R-CNN (2 classes: background + person)
# ======================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
model.to(device)


# ======================================
# 6️⃣ Training Loop
# ======================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 2
best_iou = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for imgs, tgts in train_loader:
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

        loss_dict = model(imgs, tgts)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss / len(train_loader):.4f}")

    # ======================================
    # 7️⃣ Evaluation (IoU-based mean IoU)
    # ======================================
    model.eval()
    iou_total, count = 0, 0
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs = [img.to(device) for img in imgs]
            preds = model(imgs)

            for pred, tgt in zip(preds, tgts):
                if len(pred["boxes"]) == 0 or len(tgt["boxes"]) == 0:
                    continue
                ious = ops.box_iou(pred["boxes"].cpu(), tgt["boxes"])
                max_iou, _ = ious.max(dim=1)
                iou_total += max_iou.mean().item()
                count += 1

    mean_iou = iou_total / max(count, 1)
    print(f"Validation mean IoU: {mean_iou:.4f}")

    # Save best model
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "best_pedestrian_frcnn.pth")
        print("✅ Saved best model checkpoint!")


# ======================================
# 8️⃣ Visualization
# ======================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_predictions(model, dataset, device, num_images=3, score_thresh=0.6):
    model.eval()
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        img, _ = dataset[i]
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        img_np = img.permute(1, 2, 0).numpy()
        ax = axs[i]
        ax.imshow(img_np)

        for box, score in zip(pred["boxes"], pred["scores"]):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color='yellow', fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ======================================
# 9️⃣ Run Visualization
# ======================================
visualize_predictions(model, val_ds, device)



