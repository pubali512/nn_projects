# ============================================================
# PARAMETERIZED CLASS-BASED FASTER R-CNN (COCO PERSON CLASS)
# ============================================================

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
import torchvision.models.detection as detection
import torchvision.transforms.functional as F
import torchvision.ops as ops
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random
import icons


# ======================================
# Detection Transform Class
# ======================================
class DetectionTransform:
    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)
        if self.train and random.random() < 0.5:
            width = image.shape[-1]
            image = F.hflip(image)
            if "boxes" in target:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


# ======================================
# COCO Pedestrian Dataset
# ======================================
class CocoPedestrian(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)
        anno = [a for a in anno if a["category_id"] == 1]   # Category ID = 1 is 'person' in COCO

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

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target


# ======================================
# Person Detector Class
# ======================================
class PersonDetector:
    def __init__(
        self,
        train_root,
        train_ann,
        val_root,
        val_ann,
        train_start,
        train_end,
        val_start,
        val_end,
        num_train_images,
        num_epochs,
        batch_size,
        lr,
    ):
        self.train_root = train_root
        self.train_ann = train_ann
        self.val_root = val_root
        self.val_ann = val_ann
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.num_train_images = num_train_images
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.accuracy_log = []                    # To log accuracy for 3D plotting: (epoch, num_train_images, mean_iou)
        self.loss_log = []                        # To log loss for 3D plotting: (epoch, num_train_images, avg_loss)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Set device

        print(icons.device(f" Using device: {self.device}"))
        print(icons.start(f" Training for {self.num_epochs} epochs for range ({self.train_start} - {self.train_end}) of images."))

        # Prepare everything
        self._prepare_data()
        self._prepare_model()

    # -------------------------------------
    def _prepare_data(self):
        train_ds_full = CocoPedestrian(self.train_root, self.train_ann, DetectionTransform(train=True))
        val_ds_full = CocoPedestrian(self.val_root, self.val_ann, DetectionTransform(train=False))

        # Subsets for parameterized control
        
        train_indices = list(range(
        max(0, self.train_start),
        min(self.train_end, len(train_ds_full))
        ))

        val_indices = list(range(
        max(0, self.val_start),
        min(self.val_end, len(val_ds_full))
        ))


        self.train_ds = Subset(train_ds_full, train_indices)
        self.val_ds = Subset(val_ds_full, val_indices)

        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=0
        )

    # -------------------------------------
    def _prepare_model(self):
        model = detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005
        )

    # -------------------------------------
    def train(self):
        best_iou = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for imgs, tgts in self.train_loader:
                imgs = [img.to(self.device) for img in imgs]
                tgts = [{k: v.to(self.device) for k, v in t.items()} for t in tgts]

                loss_dict = self.model(imgs, tgts)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                total_loss += losses.item()

            avg_loss = total_loss / len(self.train_loader)
            print(icons.steps(f" Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_loss:.4f}"))

            mean_iou = self.evaluate()
            
            # Log accuracy for 3D plotting
            self.accuracy_log.append(
            (epoch + 1, self.num_train_images, mean_iou)
            )
            
            # Log loss for 3D plotting
            self.loss_log.append(
            (epoch + 1, self.num_train_images, avg_loss)
            )

            
            # Update best IoU
            if mean_iou > best_iou:
                best_iou = mean_iou
                
            
    
                
        print(icons.check(f" Training complete. Best IoU: {best_iou:.5f}"))
        

    # -------------------------------------
    def evaluate(self):
        self.model.eval()
        iou_total, count = 0, 0
        with torch.no_grad():
            for imgs, tgts in self.val_loader:
                imgs = [img.to(self.device) for img in imgs]
                preds = self.model(imgs)
                for pred, tgt in zip(preds, tgts):
                    if len(pred["boxes"]) == 0 or len(tgt["boxes"]) == 0:
                        continue
                    ious = ops.box_iou(pred["boxes"].cpu(), tgt["boxes"])
                    max_iou, _ = ious.max(dim=1)
                    iou_total += max_iou.mean().item()
                    count += 1

        mean_iou = iou_total / max(count, 1)
        print(icons.info(f" Validation mean IoU: {mean_iou:.5f}"))
        return mean_iou

    # -------------------------------------
    def visualize(self, num_images, score_thresh):
        self.model.eval()
        _, axs = plt.subplots(1, num_images, figsize=(10, 5))
        for i in range(num_images):
            img, _ = self.val_ds[i]
            with torch.no_grad():
                pred = self.model([img.to(self.device)])[0]
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
        
        # Adjust layout and show    
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
    # -------------------------------------
    def plot_3d_accuracy(self, accuracy_log):
        epochs = np.array([x[0] for x in accuracy_log])
        num_images = np.array([x[1] for x in accuracy_log])
        accuracy = np.array([x[2] for x in accuracy_log])

        # Create a figure for 3D plotting
        fig = plt.figure(figsize=(10, 7))
        
        # Add 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        ax.scatter(epochs, num_images, accuracy, c=accuracy, cmap='viridis', s=60)

        # Set labels and title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of Training Images")
        ax.set_zlabel("Mean IoU (Accuracy)")
        ax.set_title("3D Accuracy – Person Detection")

        # Show the plot
        plt.show()  
        
    
    # ------------------------------------- 
    def plot_3d_loss(self,loss_log): 
        epochs = np.array([x[0] for x in loss_log])
        num_images = np.array([x[1] for x in loss_log])
        loss = np.array([x[2] for x in loss_log])

        # Create a figure for 3D plotting
        fig = plt.figure(figsize=(10, 7))
        
        # Add 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        ax.scatter(epochs, num_images, loss, c=loss, cmap='plasma', s=60)

        # Set labels and title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of Training Images")
        ax.set_zlabel("Training Loss")
        ax.set_title("3D Training Loss – Person Detection")

        # Show the plot
        plt.show()  
        

# ======================================
# Main
# ======================================
if __name__ == "__main__":
    detector = PersonDetector(
        train_root="/Users/pubalimazumder/datasets/coco/train2017",
        train_ann="/Users/pubalimazumder/datasets/coco/annotations/instances_train2017.json",
        val_root="/Users/pubalimazumder/datasets/coco/val2017",
        val_ann="/Users/pubalimazumder/datasets/coco/annotations/instances_val2017.json",
        train_start=2000,
        train_end=2100,
        val_start=500,
        val_end=600,
        num_train_images=200,
        num_epochs=6,
        batch_size=2,
        lr=0.005,
    )

    detector.train()
    detector.visualize(num_images=6, score_thresh=0.6)
    detector.plot_3d_accuracy(detector.accuracy_log)
    detector.plot_3d_loss(detector.loss_log)
