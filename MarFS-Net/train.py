#论文标准的train函数
import os
import sys

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import random
import numpy as np
import pandas as pd  # 🟢 【修改1】：引入 pandas 用于保存 CSV 记录
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import time
import warnings
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip,
    Normalize, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue
)
from albumentations.pytorch import ToTensorV2
from PIL import Image

if hasattr(torch, '_dynamo'):
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True

import archs
import losses
from metrics import indicators
from utils import AverageMeter

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 配置区域 =================
config = {
    'name': 'UNet_Classic',
    'epochs': 300,

    # 🟢 【提速与显存优化 1】：分辨率降到经典的 256x256
    'input_h': 256,
    'input_w': 256,

    'batch_size': 4,
    'accumulation_steps': 2,
    'num_workers': 4,
    'dataseed': 3407,

    'arch': 'UNet_Classic',
    'deep_supervision': True,
    'input_channels': 3,
    'num_classes': 8,

    # 🟢 【提速与显存优化 2】：通道数减半，完美抵消 stride=1 带来的空间翻倍
    'input_list': [32, 64, 128],

    'loss': 'DiceCELoss',
    'optimizer': 'AdamW',

    # 🟢 【大提速】：重新开启 AMP 混合精度！因为在 arch 里加了安全层，现在不会 NaN 了
    'amp': True,

    'max_lr': 1e-3,
    'weight_decay': 1e-2,

    'dataset': 'custom',
    'data_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs',
    'output_dir': 'outputs',

    'img_ext': '.png',
    'mask_ext': '.png',
    'HorizontalFlip': 0.5,
    'multi_scale_training': False
}


class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform
        self.MASK_SUFFIX = "_pure_mask_single"

        self.real_img_files = {f.lower(): f for f in os.listdir(img_dir) if
                               f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
        self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir) if
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}

        print(f"⚡ Verifying image paths in: {img_dir}")
        self.valid_data = []

        for img_id in tqdm(img_ids, desc="Scanning Paths"):
            img_name = None
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                test_name = (img_id + ext).lower()
                if test_name in self.real_img_files:
                    img_name = self.real_img_files[test_name]
                    break
            if img_name is None: continue

            mask_name = None
            target_mask_base = img_id + self.MASK_SUFFIX
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                test_name = (target_mask_base + ext).lower()
                if test_name in self.real_mask_files:
                    mask_name = self.real_mask_files[test_name]
                    break
            if mask_name is None:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    test_name = (img_id + ext).lower()
                    if test_name in self.real_mask_files:
                        mask_name = self.real_mask_files[test_name]
                        break
            if mask_name is None: continue

            self.valid_data.append({
                'id': img_id,
                'img_path': os.path.join(self.img_dir, img_name),
                'mask_path': os.path.join(self.mask_dir, mask_name)
            })

        print(f"✅ Scanning Complete. Found: {len(self.valid_data)} valid pairs.")
        if len(self.valid_data) == 0:
            raise RuntimeError("❌ 没有匹配到数据！请检查 inputs 文件夹路径结构。")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        data_info = self.valid_data[idx]

        img_obj = Image.open(data_info['img_path']).convert('RGB')
        image = np.array(img_obj)

        mask_obj = Image.open(data_info['mask_path']).convert('L')
        mask = np.array(mask_obj)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            image = cv2.resize(image, (config['input_w'], config['input_h']))
            mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
            image = image.astype('float32') / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask).long()

        return image, mask.long(), data_info['id']


def train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader), desc=f"Ep {epoch + 1} Train", leave=True)
    accum_steps = config.get('accumulation_steps', 1)

    current_weights = [1.0, 0.2, 0.3, 0.3, 0.5]
    if epoch > config['epochs'] * 0.8: current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]

    optimizer.zero_grad()

    for i, (input, target, _) in enumerate(train_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(enabled=config['amp']):
            outputs = model(input)

            if config['deep_supervision'] and isinstance(outputs, (list, tuple)):
                loss = 0
                for idx, o in enumerate(outputs):
                    if idx >= len(current_weights) or current_weights[idx] == 0: continue
                    if o.shape[2:] != target.shape[1:]:
                        o = F.interpolate(o, size=target.shape[1:], mode='bilinear', align_corners=False)
                    loss += current_weights[idx] * criterion(o, target)
                final_output = outputs[0]
            else:
                loss = criterion(outputs, target)
                final_output = outputs

            loss = loss / accum_steps

        if torch.isnan(loss):
            print(f"⚠️ Warning: Loss is NaN at step {i}. Skipping batch.")
            optimizer.zero_grad()
            continue

        if config['amp'] and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0:
            if config['amp'] and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            ema_model.update_parameters(model)
            scheduler.step()

        with torch.no_grad():
            if final_output.shape[2:] != target.shape[1:]:
                final_output_metric = F.interpolate(final_output, size=target.shape[1:], mode='bilinear',
                                                    align_corners=False)
            else:
                final_output_metric = final_output
            iou, _, _, _, _, _ = indicators(final_output_metric, target, compute_hd95=False)

        avg_meters['loss'].update(loss.item() * accum_steps, input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        pbar.set_postfix({'L': f"{avg_meters['loss'].avg:.4f}", 'IoU': f"{avg_meters['iou'].avg:.4f}"})
        pbar.update(1)

    pbar.close()
    return {k: v.avg for k, v in avg_meters.items()}


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.eval()
    torch.cuda.empty_cache()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), desc="Validating", leave=False)
        for input, target, _ in val_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast(enabled=config['amp']):
                outputs = model(input)

            if isinstance(outputs, (list, tuple)):
                final_output = outputs[0]
            else:
                final_output = outputs

            if final_output.shape[2:] != target.shape[1:]:
                final_output = F.interpolate(final_output, size=target.shape[1:], mode='bilinear', align_corners=False)

            # 为了计算稳定，把输出转回 fp32
            final_output = final_output.float()

            loss = criterion(final_output, target)
            iou, _, _, _, _, _ = indicators(final_output, target, compute_hd95=False)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            pbar.update(1)
        pbar.close()
    return {k: v.avg for k, v in avg_meters.items()}


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f"✅ Device: {device}")
    print(
        f"🚀 Mode: RTX 5060 AMP Fast (Workers: {config['num_workers']}, Res: {config['input_w']}, AMP: {config['amp']})")

    base_dir = config['data_dir']
    dataset_name = config['dataset']
    images_dir = os.path.join(base_dir, dataset_name, 'images')
    masks_dir = os.path.join(base_dir, dataset_name, 'masks')

    all_files_in_dir = os.listdir(images_dir)
    valid_ids = [os.path.splitext(f)[0] for f in all_files_in_dir if
                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"   -> 扫描到 {len(valid_ids)} 张图片ID")

    # 🟢 【修改2】：将 test_size=0.2 改为了 0.3，实现 7:3 划分
    train_ids, val_ids = train_test_split(valid_ids, test_size=0.3, random_state=config['dataseed'])
    print(f"   -> 数据划分: 训练集 {len(train_ids)} 张, 测试集 {len(val_ids)} 张")

    os.makedirs(config['output_dir'], exist_ok=True)
    save_path = os.path.join(config['output_dir'], config['name'])
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        embed_dims=config['input_list']
    ).to(device)

    decay = 0.999
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
        decay * averaged_model_parameter + (1.0 - decay) * model_parameter
    ema_model = AveragedModel(model, avg_fn=ema_avg)

    try:
        criterion = losses.__dict__[config['loss']](num_classes=config['num_classes'], label_smoothing=0.1).to(device)
    except:
        criterion = losses.__dict__[config['loss']](num_classes=config['num_classes']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler() if config['amp'] else None

    print("\n=> Preparing Data...")

    train_tf = Compose([
        Resize(config['input_h'], config['input_w']),
        HorizontalFlip(p=config.get('HorizontalFlip', 0.5)),
        VerticalFlip(p=0.5),
        Normalize(), ToTensorV2()
    ])
    val_tf = Compose([Resize(config['input_h'], config['input_w']), Normalize(), ToTensorV2()])

    train_ds = LazyDataset(train_ids, images_dir, masks_dir, config['num_classes'], train_tf)
    val_ds = LazyDataset(val_ids, images_dir, masks_dir, config['num_classes'], val_tf)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=(config['num_workers'] > 0)
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=(config['num_workers'] > 0)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['max_lr'], epochs=config['epochs'],
        steps_per_epoch=len(train_loader) // config['accumulation_steps'],
        pct_start=0.3, div_factor=25, final_div_factor=10000, anneal_strategy='cos'
    )

    best_iou = 0
    print(f"\n🚀 Start Training... (Validating every epoch and logging to CSV)")

    # 🟢 【修改3】：初始化记录列表与 CSV 保存路径
    training_log_list = []
    csv_log_path = os.path.join(save_path, 'training_log.csv')

    for epoch in range(config['epochs']):
        print(f'\nEpoch [{epoch + 1}/{config["epochs"]}]')
        start = time.time()

        train_log = train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler)

        # 🟢 【修改4】：去掉 if 限制，训练多少轮就验证多少轮
        val_log = validate(config, val_loader, ema_model, criterion)
        val_iou = val_log['iou']
        val_str = f"{val_iou:.4f}"

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(ema_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
            print(f"⭐ New Best IoU: {best_iou:.4f} (Model Saved!)")

        curr_lr = optimizer.param_groups[0]['lr']
        print(
            f"   Time: {time.time() - start:.1f}s | LR: {curr_lr:.2e} | Train IoU: {train_log['iou']:.4f} | Val IoU: {val_str}")

        # 🟢 【修改5】：将每一轮的指标存入列表并写入 CSV，保存在权重同一目录下
        training_log_list.append({
            'Epoch': epoch + 1,
            'Train_Loss': train_log['loss'],
            'Train_IoU': train_log['iou'],
            'Val_Loss': val_log['loss'],
            'Val_IoU': val_log['iou'],
            'Learning_Rate': curr_lr
        })
        pd.DataFrame(training_log_list).to_csv(csv_log_path, index=False)


if __name__ == '__main__':
    main()
