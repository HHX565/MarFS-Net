import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset
from glob import glob


class Dataset(BaseDataset):
    def __init__(self, ids, images_dir, masks_dir, img_ext, mask_ext, num_classes, transform=None):
        self.ids = ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.checked_mask_format = False

    def _find_mask_path(self, img_id):
        # === [关键修复] 补全所有可能的标签后缀 ===
        candidates = [
            img_id + self.mask_ext,  # 图片名.png
            img_id + '.png',  # 强制 .png
            img_id + '_mask.png',  # _mask.png
            img_id + '_L.png',  # _L.png
            img_id + '_pure_mask_single.png',  # <--- [找回] 你的标签格式
            img_id + '_pure_mask_single.jpg',  # 容错 jpg
            img_id + '_label.png'  # 常见格式
        ]

        for c in candidates:
            path = os.path.join(self.masks_dir, c)
            if os.path.exists(path):
                return path
        return None

    def _preprocess_mask(self, mask):
        """
        智能预处理：自动判断是否需要映射
        """
        unique_vals = np.unique(mask)
        max_val = np.max(unique_vals)

        # 1. 如果已经是 0,1,2... 的索引图，直接用
        if max_val < self.num_classes:
            if not self.checked_mask_format:
                print(f"✅ [Dataset] 检测到单通道索引标签 (Max: {max_val})，跳过映射。")
                self.checked_mask_format = True
            return mask.astype(np.int64)

        # 2. 否则执行映射 (兼容旧数据)
        if not self.checked_mask_format:
            print(f"⚠️ [Dataset] 检测到原始灰度标签 (Values: {unique_vals})，执行映射。")
            self.checked_mask_format = True

        mapping = {
            0: 0,
            76: 1, 81: 2, 149: 3, 173: 4, 188: 5, 225: 6
        }

        new_mask = np.zeros_like(mask, dtype=np.int64)
        for old_val, new_val in mapping.items():
            new_mask[mask == old_val] = new_val

        return new_mask

    def __getitem__(self, i):
        img_id = self.ids[i]

        # 1. 读取图片 (容错多种后缀)
        img_path = os.path.join(self.images_dir, img_id + self.img_ext)
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.bmp']:
                temp = os.path.join(self.images_dir, img_id + ext)
                if os.path.exists(temp): img_path = temp; break

        image = cv2.imread(img_path)
        if image is None:
            # 最后的容错：如果图片坏了，跳过或报错
            raise ValueError(f"无法读取图片: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 读取 Mask
        mask_path = self._find_mask_path(img_id)
        if mask_path is None:
            raise ValueError(f"找不到 Mask: {img_id} (请检查 dataset.py 中的 candidates 列表)")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 3. 处理
        mask = self._preprocess_mask(mask)

        # 4. 增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 5. 转 Tensor
        if isinstance(image, np.ndarray):
            image = image.astype('float32') / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)

        if isinstance(mask, np.ndarray):
            mask = mask.astype('int64')
            mask = torch.from_numpy(mask)

        return image, mask, img_id

    def __len__(self):
        return len(self.ids)