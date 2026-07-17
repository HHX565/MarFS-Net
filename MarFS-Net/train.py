# import os
# import sys
#
# # ==============================================================================
# # 【核武器级补丁】强制禁用 Windows 下不支持的所有 PyTorch 编译/Triton 功能
# # 必须放在 import torch 之前设置环境变量！
# # ==============================================================================
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import cv2
#
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
#
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import yaml
# import time
# import warnings
# from glob import glob
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from torch.cuda.amp import GradScaler, autocast
# from torch.optim.swa_utils import AveragedModel
# from albumentations import (
#     Compose, Resize, HorizontalFlip, VerticalFlip,
#     Normalize, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue
# )
# from albumentations.pytorch import ToTensorV2
# from PIL import Image
#
# # 再次在代码层面强制关闭编译
# if hasattr(torch, '_dynamo'):
#     torch._dynamo.config.disable = True
#     torch._dynamo.config.suppress_errors = True
#
# import archs
# import losses
# from metrics import indicators
# from utils import AverageMeter
#
# warnings.filterwarnings("ignore")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # ================= 配置区域 (RTX 5060 8GB 特调版) =================
# config = {
#     # 实验名称
#     'name': 'my_paper',
#     'epochs': 300,
#
#     # 【显存压榨策略】
#     # 352x352 配合 BS=4
#     'input_h': 352,
#     'input_w': 352,
#
#     'batch_size': 4,
#     'accumulation_steps': 2,
#
#     'num_workers': 0,
#     'dataseed': 3407,
#
#     # 【🔥 关键修改 1】对应 archs.py 中的新类名
#     'arch': 'HDNet_Pseudo_DeepDense',
#     'deep_supervision': True,
#     'input_channels': 3,
#     'num_classes': 8,
#
#     # 【🔥 关键修改 2】维度适配
#     # HDNet_UKAN_Dense 是 3 层结构，所以这里只需要 3 个参数
#     # 使用 [64, 128, 256] 是最稳健的配置
#     'input_list': [64, 128, 256],
#
#     'loss': 'DiceCELoss',
#     'optimizer': 'AdamW',
#
#     'amp': True,
#
#     # 【🔥 关键修改 3】降低学习率以适应 KAN+Dense 的复杂梯度
#     'max_lr': 1e-3,
#     'weight_decay': 1e-2,
#
#     # 路径配置
#     'dataset': 'custom',
#     'data_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs',
#     'output_dir': 'outputs',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#
#     'cutmix_prob': 0.3,
#     'multi_scale_training': False
# }
#
#
# # ================= 智能 Dataset (PIL版) =================
# class RamCachedDataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None):
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.num_classes = num_classes
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir) if
#                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir) if
#                                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#
#         print(f"⚡ Pre-loading images into RAM from: {img_dir}")
#         self.data_cache = []
#         valid_ids = []
#
#         for img_id in tqdm(img_ids, desc="Caching"):
#             img_name = None
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (img_id + ext).lower()
#                 if test_name in self.real_img_files:
#                     img_name = self.real_img_files[test_name]
#                     break
#             if img_name is None: continue
#
#             mask_name = None
#             target_mask_base = img_id + self.MASK_SUFFIX
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (target_mask_base + ext).lower()
#                 if test_name in self.real_mask_files:
#                     mask_name = self.real_mask_files[test_name]
#                     break
#             if mask_name is None:
#                 for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                     test_name = (img_id + ext).lower()
#                     if test_name in self.real_mask_files:
#                         mask_name = self.real_mask_files[test_name]
#                         break
#             if mask_name is None: continue
#
#             try:
#                 img_path = os.path.join(self.img_dir, img_name)
#                 mask_path = os.path.join(self.mask_dir, mask_name)
#
#                 img_obj = Image.open(img_path).convert('RGB')
#                 image = np.array(img_obj)
#
#                 mask_obj = Image.open(mask_path).convert('L')
#                 mask = np.array(mask_obj)
#
#                 self.data_cache.append({'image': image, 'mask': mask})
#                 valid_ids.append(img_id)
#             except Exception:
#                 continue
#
#         self.img_ids = valid_ids
#         print(f"✅ Caching Complete. Loaded: {len(self.img_ids)} pairs.")
#         if len(self.img_ids) == 0:
#             raise RuntimeError("❌ 没有匹配到数据！请检查 inputs 文件夹路径结构。")
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         data = self.data_cache[idx]
#         image, mask = data['image'], data['mask']
#
#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image, mask = augmented['image'], augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask.long(), self.img_ids[idx]
#
#
# def train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.train()
#     pbar = tqdm(total=len(train_loader), desc=f"Ep {epoch + 1} Train", leave=True)
#     accum_steps = config.get('accumulation_steps', 1)
#
#     # 对应 archs.py 中 forward 返回的 5 个输出 [final, out1, out2, out3, out4]
#     current_weights = [1.0, 0.4, 0.3, 0.2, 0.1]
#
#     if epoch > config['epochs'] * 0.8: current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
#
#     optimizer.zero_grad()
#
#     for i, (input, target, _) in enumerate(train_loader):
#         input = input.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         with autocast(enabled=config['amp']):
#             outputs = model(input)
#
#             if config['deep_supervision'] and isinstance(outputs, (list, tuple)):
#                 loss = 0
#                 for idx, o in enumerate(outputs):
#                     if idx >= len(current_weights) or current_weights[idx] == 0: continue
#                     if o.shape[2:] != target.shape[1:]:
#                         o = F.interpolate(o, size=target.shape[1:], mode='bilinear', align_corners=False)
#                     loss += current_weights[idx] * criterion(o, target)
#                 final_output = outputs[0]
#             else:
#                 loss = criterion(outputs, target)
#                 final_output = outputs
#
#             loss = loss / accum_steps
#
#         # 增加 NaN 报警
#         if torch.isnan(loss):
#             print(f"⚠️ Warning: Loss is NaN at step {i}. Skipping batch.")
#             optimizer.zero_grad()
#             continue
#
#         if config['amp'] and scaler is not None:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()
#
#         if (i + 1) % accum_steps == 0:
#             if config['amp'] and scaler is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#             optimizer.zero_grad()
#             ema_model.update_parameters(model)
#             scheduler.step()
#
#         with torch.no_grad():
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output_metric = F.interpolate(final_output, size=target.shape[1:], mode='bilinear',
#                                                     align_corners=False)
#             else:
#                 final_output_metric = final_output
#             iou, _, _, _, _, _ = indicators(final_output_metric, target, compute_hd95=False)
#
#         avg_meters['loss'].update(loss.item() * accum_steps, input.size(0))
#         avg_meters['iou'].update(iou, input.size(0))
#         pbar.set_postfix({'L': f"{avg_meters['loss'].avg:.4f}", 'IoU': f"{avg_meters['iou'].avg:.4f}"})
#         pbar.update(1)
#
#     pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def validate(config, val_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.eval()
#     torch.cuda.empty_cache()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(val_loader), desc="Validating", leave=False)
#         for input, target, _ in val_loader:
#             input = input.to(device, non_blocking=True)
#             target = target.to(device, non_blocking=True)
#             outputs = model(input)
#
#             if isinstance(outputs, (list, tuple)):
#                 final_output = outputs[0]
#             else:
#                 final_output = outputs
#
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output = F.interpolate(final_output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             loss = criterion(final_output, target)
#             iou, _, _, _, _, _ = indicators(final_output, target, compute_hd95=False)
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#             pbar.update(1)
#         pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def main():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#
#     print(f"✅ Device: {device}")
#     print("🚀 Mode: RTX 5060 FAST (AMP + Light Model + BS=4)")
#
#     base_dir = config['data_dir']
#     dataset_name = config['dataset']
#     images_dir = os.path.join(base_dir, dataset_name, 'images')
#     masks_dir = os.path.join(base_dir, dataset_name, 'masks')
#
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [os.path.splitext(f)[0] for f in all_files_in_dir if
#                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     print(f"   -> 扫描到 {len(valid_ids)} 张图片ID")
#     train_ids, val_ids = train_test_split(valid_ids, test_size=0.2, random_state=config['dataseed'])
#
#     os.makedirs(config['output_dir'], exist_ok=True)
#     save_path = os.path.join(config['output_dir'], config['name'])
#     os.makedirs(save_path, exist_ok=True)
#     with open(os.path.join(save_path, 'config.yml'), 'w') as f:
#         yaml.dump(config, f)
#
#     # 初始化模型
#     model = archs.__dict__[config['arch']](
#         num_classes=config['num_classes'],
#         input_channels=config['input_channels'],
#         deep_supervision=config['deep_supervision'],
#         embed_dims=config['input_list']
#     ).to(device)
#
#     decay = 0.999
#     ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
#         decay * averaged_model_parameter + (1.0 - decay) * model_parameter
#     ema_model = AveragedModel(model, avg_fn=ema_avg)
#
#     try:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes'], label_smoothing=0.1).to(device)
#     except:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes']).to(device)
#
#     optimizer = optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])
#     scaler = torch.cuda.amp.GradScaler() if config['amp'] else None
#
#     print("\n=> Preparing Data...")
#
#     train_tf = Compose([
#         Resize(config['input_h'], config['input_w']),
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         Normalize(), ToTensorV2()
#     ])
#     val_tf = Compose([Resize(config['input_h'], config['input_w']), Normalize(), ToTensorV2()])
#
#     train_ds = RamCachedDataset(train_ids, images_dir, masks_dir, config['num_classes'], train_tf)
#     val_ds = RamCachedDataset(val_ids, images_dir, masks_dir, config['num_classes'], val_tf)
#
#     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0,
#                                                drop_last=True)
#     val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
#
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=config['max_lr'], epochs=config['epochs'],
#         steps_per_epoch=len(train_loader) // config['accumulation_steps'],
#         pct_start=0.3, div_factor=25, final_div_factor=10000, anneal_strategy='cos'
#     )
#
#     best_iou = 0
#     print(f"\n🚀 Start Training... (Val every 10 epochs)")
#
#     for epoch in range(config['epochs']):
#         print(f'\nEpoch [{epoch + 1}/{config["epochs"]}]')
#         start = time.time()
#
#         train_log = train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler)
#
#         val_str = "Skipped"
#
#         if (epoch + 1) % 10 == 0 or epoch > config['epochs'] - 20:
#             val_log = validate(config, val_loader, ema_model, criterion)
#             val_iou = val_log['iou']
#             val_str = f"{val_iou:.4f}"
#             if val_iou > best_iou:
#                 best_iou = val_iou
#                 torch.save(ema_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
#                 print(f"⭐ New Best IoU: {best_iou:.4f}")
#
#         curr_lr = optimizer.param_groups[0]['lr']
#         print(
#             f"   Time: {time.time() - start:.1f}s | LR: {curr_lr:.2e} | Train IoU: {train_log['iou']:.4f} | Val IoU: {val_str}")
#
#
# if __name__ == '__main__':
#     main()







#import os
import sys

# ==============================================================================
# 【核武器级补丁】强制禁用 Windows 下不支持的所有 PyTorch 编译/Triton 功能
# 必须放在 import torch 之前设置环境变量！
# ==============================================================================
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#import cv2

# 极度关键：防止 OpenCV 自带的多线程和 PyTorch 的 num_workers 多进程死锁
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
#
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import yaml
# import time
# import warnings
# from glob import glob
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from torch.cuda.amp import GradScaler, autocast
# from torch.optim.swa_utils import AveragedModel
# from albumentations import (
#     Compose, Resize, HorizontalFlip, VerticalFlip,
#     Normalize, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue
# )
# from albumentations.pytorch import ToTensorV2
# from PIL import Image
#
# # 再次在代码层面强制关闭编译
# if hasattr(torch, '_dynamo'):
#     torch._dynamo.config.disable = True
#     torch._dynamo.config.suppress_errors = True
#
# import archs
# import losses
# from metrics import indicators
# from utils import AverageMeter
#
# warnings.filterwarnings("ignore")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # ================= 配置区域 =================
# config = {
#     'name': 'my_paper',
#     'epochs': 300,
#
#     # 如果 archs.py 中使用了 stride=1，建议减小尺寸或减小 BS，否则 8G 显存会爆
#     'input_h': 352,
#     'input_w': 352,
#
#     'batch_size': 4,
#     'accumulation_steps': 2,
#
#     # 🟢 【多进程提速核心】：4 是大多数电脑的黄金平衡点
#     'num_workers': 4,
#     'dataseed': 3407,
#
#     'arch': 'HDNet_Pseudo_DeepDense',
#     'deep_supervision': True,
#     'input_channels': 3,
#     'num_classes': 8,
#
#     'input_list': [64, 128, 256],
#
#     'loss': 'DiceCELoss',
#     'optimizer': 'AdamW',
#
#     # 保持关闭，防止 KAN 数值爆炸
#     'amp': False,
#
#     'max_lr': 1e-3,
#     'weight_decay': 1e-2,
#
#     'dataset': 'custom',
#     'data_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs',
#     'output_dir': 'outputs',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#
#     'HorizontalFlip': 0.5,
#     'multi_scale_training': False
# }
#
#
# # ================= 标准流式加载 Dataset (Lazy Loading) =================
# class LazyDataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None):
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.num_classes = num_classes
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir) if
#                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir) if
#                                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#
#         print(f"⚡ Verifying image paths in: {img_dir}")
#         self.valid_data = []
#
#         # 扫描阶段：只存路径
#         for img_id in tqdm(img_ids, desc="Scanning Paths"):
#             img_name = None
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (img_id + ext).lower()
#                 if test_name in self.real_img_files:
#                     img_name = self.real_img_files[test_name]
#                     break
#             if img_name is None: continue
#
#             mask_name = None
#             target_mask_base = img_id + self.MASK_SUFFIX
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (target_mask_base + ext).lower()
#                 if test_name in self.real_mask_files:
#                     mask_name = self.real_mask_files[test_name]
#                     break
#             if mask_name is None:
#                 for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                     test_name = (img_id + ext).lower()
#                     if test_name in self.real_mask_files:
#                         mask_name = self.real_mask_files[test_name]
#                         break
#             if mask_name is None: continue
#
#             self.valid_data.append({
#                 'id': img_id,
#                 'img_path': os.path.join(self.img_dir, img_name),
#                 'mask_path': os.path.join(self.mask_dir, mask_name)
#             })
#
#         print(f"✅ Scanning Complete. Found: {len(self.valid_data)} valid pairs.")
#         if len(self.valid_data) == 0:
#             raise RuntimeError("❌ 没有匹配到数据！请检查 inputs 文件夹路径结构。")
#
#     def __len__(self):
#         return len(self.valid_data)
#
#     def __getitem__(self, idx):
#         # 🟢 流式读取：多进程调度到这个样本时，才去硬盘发生真正的 I/O 操作
#         data_info = self.valid_data[idx]
#
#         img_obj = Image.open(data_info['img_path']).convert('RGB')
#         image = np.array(img_obj)
#
#         mask_obj = Image.open(data_info['mask_path']).convert('L')
#         mask = np.array(mask_obj)
#
#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image, mask = augmented['image'], augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask.long(), data_info['id']
#
#
# def train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.train()
#     pbar = tqdm(total=len(train_loader), desc=f"Ep {epoch + 1} Train", leave=True)
#     accum_steps = config.get('accumulation_steps', 1)
#
#     current_weights = [1.0, 0.4, 0.3, 0.2, 0.1]
#     if epoch > config['epochs'] * 0.8: current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
#
#     optimizer.zero_grad()
#
#     for i, (input, target, _) in enumerate(train_loader):
#         input = input.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         with autocast(enabled=config['amp']):
#             outputs = model(input)
#
#             if config['deep_supervision'] and isinstance(outputs, (list, tuple)):
#                 loss = 0
#                 for idx, o in enumerate(outputs):
#                     if idx >= len(current_weights) or current_weights[idx] == 0: continue
#                     if o.shape[2:] != target.shape[1:]:
#                         o = F.interpolate(o, size=target.shape[1:], mode='bilinear', align_corners=False)
#                     loss += current_weights[idx] * criterion(o, target)
#                 final_output = outputs[0]
#             else:
#                 loss = criterion(outputs, target)
#                 final_output = outputs
#
#             loss = loss / accum_steps
#
#         if torch.isnan(loss):
#             print(f"⚠️ Warning: Loss is NaN at step {i}. Skipping batch.")
#             optimizer.zero_grad()
#             continue
#
#         if config['amp'] and scaler is not None:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()
#
#         if (i + 1) % accum_steps == 0:
#             if config['amp'] and scaler is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#             optimizer.zero_grad()
#             ema_model.update_parameters(model)
#             scheduler.step()
#
#         with torch.no_grad():
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output_metric = F.interpolate(final_output, size=target.shape[1:], mode='bilinear',
#                                                     align_corners=False)
#             else:
#                 final_output_metric = final_output
#             iou, _, _, _, _, _ = indicators(final_output_metric, target, compute_hd95=False)
#
#         avg_meters['loss'].update(loss.item() * accum_steps, input.size(0))
#         avg_meters['iou'].update(iou, input.size(0))
#         pbar.set_postfix({'L': f"{avg_meters['loss'].avg:.4f}", 'IoU': f"{avg_meters['iou'].avg:.4f}"})
#         pbar.update(1)
#
#     pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def validate(config, val_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.eval()
#     torch.cuda.empty_cache()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(val_loader), desc="Validating", leave=False)
#         for input, target, _ in val_loader:
#             input = input.to(device, non_blocking=True)
#             target = target.to(device, non_blocking=True)
#             outputs = model(input)
#
#             if isinstance(outputs, (list, tuple)):
#                 final_output = outputs[0]
#             else:
#                 final_output = outputs
#
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output = F.interpolate(final_output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             loss = criterion(final_output, target)
#             iou, _, _, _, _, _ = indicators(final_output, target, compute_hd95=False)
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#             pbar.update(1)
#         pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def main():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#
#     print(f"✅ Device: {device}")
#     print(f"🚀 Mode: RTX 5060 FAST (Multi-Processing Workers: {config['num_workers']})")
#
#     base_dir = config['data_dir']
#     dataset_name = config['dataset']
#     images_dir = os.path.join(base_dir, dataset_name, 'images')
#     masks_dir = os.path.join(base_dir, dataset_name, 'masks')
#
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [os.path.splitext(f)[0] for f in all_files_in_dir if
#                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     print(f"   -> 扫描到 {len(valid_ids)} 张图片ID")
#     train_ids, val_ids = train_test_split(valid_ids, test_size=0.2, random_state=config['dataseed'])
#
#     os.makedirs(config['output_dir'], exist_ok=True)
#     save_path = os.path.join(config['output_dir'], config['name'])
#     os.makedirs(save_path, exist_ok=True)
#     with open(os.path.join(save_path, 'config.yml'), 'w') as f:
#         yaml.dump(config, f)
#
#     # 初始化模型
#     model = archs.__dict__[config['arch']](
#         num_classes=config['num_classes'],
#         input_channels=config['input_channels'],
#         deep_supervision=config['deep_supervision'],
#         embed_dims=config['input_list']
#     ).to(device)
#
#     decay = 0.999
#     ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
#         decay * averaged_model_parameter + (1.0 - decay) * model_parameter
#     ema_model = AveragedModel(model, avg_fn=ema_avg)
#
#     try:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes'], label_smoothing=0.1).to(device)
#     except:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes']).to(device)
#
#     optimizer = optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])
#     scaler = torch.cuda.amp.GradScaler() if config['amp'] else None
#
#     print("\n=> Preparing Data...")
#
#     train_tf = Compose([
#         Resize(config['input_h'], config['input_w']),
#         HorizontalFlip(p=config.get('HorizontalFlip', 0.5)),
#         VerticalFlip(p=0.5),
#         Normalize(), ToTensorV2()
#     ])
#     val_tf = Compose([Resize(config['input_h'], config['input_w']), Normalize(), ToTensorV2()])
#
#     train_ds = LazyDataset(train_ids, images_dir, masks_dir, config['num_classes'], train_tf)
#     val_ds = LazyDataset(val_ids, images_dir, masks_dir, config['num_classes'], val_tf)
#
#     # 🟢 【多进程数据流核心装配】
#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         pin_memory=True,  # 开启锁页内存，加速 CPU 到 GPU 的拷贝
#         drop_last=True,
#         persistent_workers=(config['num_workers'] > 0)  # 保持进程存活，Epoch 切换不卡顿
#     )
#
#     val_loader = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         persistent_workers=(config['num_workers'] > 0)
#     )
#
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=config['max_lr'], epochs=config['epochs'],
#         steps_per_epoch=len(train_loader) // config['accumulation_steps'],
#         pct_start=0.3, div_factor=25, final_div_factor=10000, anneal_strategy='cos'
#     )
#
#     best_iou = 0
#     print(f"\n🚀 Start Training... (Val every 10 epochs)")
#
#     for epoch in range(config['epochs']):
#         print(f'\nEpoch [{epoch + 1}/{config["epochs"]}]')
#         start = time.time()
#
#         train_log = train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler)
#
#         val_str = "Skipped"
#
#         if (epoch + 1) % 10 == 0 or epoch > config['epochs'] - 20:
#             val_log = validate(config, val_loader, ema_model, criterion)
#             val_iou = val_log['iou']
#             val_str = f"{val_iou:.4f}"
#             if val_iou > best_iou:
#                 best_iou = val_iou
#                 torch.save(ema_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
#                 print(f"⭐ New Best IoU: {best_iou:.4f}")
#
#         curr_lr = optimizer.param_groups[0]['lr']
#         print(
#             f"   Time: {time.time() - start:.1f}s | LR: {curr_lr:.2e} | Train IoU: {train_log['iou']:.4f} | Val IoU: {val_str}")
#
#
# # Windows 下多进程必须包在 __main__ 里面，这部分你原本就写得非常标准
# if __name__ == '__main__':
#     main()



#我论文的标准train函数
# import os
# import sys
#
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import cv2
#
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
#
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import yaml
# import time
# import warnings
# from glob import glob
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from torch.cuda.amp import GradScaler, autocast
# from torch.optim.swa_utils import AveragedModel
# from albumentations import (
#     Compose, Resize, HorizontalFlip, VerticalFlip,
#     Normalize, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue
# )
# from albumentations.pytorch import ToTensorV2
# from PIL import Image
#
# if hasattr(torch, '_dynamo'):
#     torch._dynamo.config.disable = True
#     torch._dynamo.config.suppress_errors = True
#
# import archs
# import losses
# from metrics import indicators
# from utils import AverageMeter
#
# warnings.filterwarnings("ignore")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # ================= 配置区域 =================
# config = {
#     'name': 'UKAN_OSGA_3',
#     'epochs': 300,
#
#     # 🟢 【提速与显存优化 1】：分辨率降到经典的 256x256
#     'input_h': 256,
#     'input_w': 256,
#
#     'batch_size': 4,
#     'accumulation_steps': 2,
#     'num_workers': 4,
#     'dataseed': 3407,
#
#     'arch': 'UKAN_Baseline',
#     'deep_supervision': False,
#     'input_channels': 3,
#     'num_classes': 8,
#
#     # 🟢 【提速与显存优化 2】：通道数减半，完美抵消 stride=1 带来的空间翻倍
#     'input_list': [32, 64, 128],
#
#     'loss': 'DiceCELoss',
#     'optimizer': 'AdamW',
#
#     # 🟢 【大提速】：重新开启 AMP 混合精度！因为在 arch 里加了安全层，现在不会 NaN 了
#     'amp': True,
#
#     'max_lr': 1e-3,
#     'weight_decay': 1e-2,
#
#     'dataset': 'custom',
#     'data_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs',
#     'output_dir': 'outputs',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#     'HorizontalFlip': 0.5,
#     'multi_scale_training': False
# }
#
#
# class LazyDataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None):
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.num_classes = num_classes
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir) if
#                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir) if
#                                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#
#         print(f"⚡ Verifying image paths in: {img_dir}")
#         self.valid_data = []
#
#         for img_id in tqdm(img_ids, desc="Scanning Paths"):
#             img_name = None
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (img_id + ext).lower()
#                 if test_name in self.real_img_files:
#                     img_name = self.real_img_files[test_name]
#                     break
#             if img_name is None: continue
#
#             mask_name = None
#             target_mask_base = img_id + self.MASK_SUFFIX
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (target_mask_base + ext).lower()
#                 if test_name in self.real_mask_files:
#                     mask_name = self.real_mask_files[test_name]
#                     break
#             if mask_name is None:
#                 for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                     test_name = (img_id + ext).lower()
#                     if test_name in self.real_mask_files:
#                         mask_name = self.real_mask_files[test_name]
#                         break
#             if mask_name is None: continue
#
#             self.valid_data.append({
#                 'id': img_id,
#                 'img_path': os.path.join(self.img_dir, img_name),
#                 'mask_path': os.path.join(self.mask_dir, mask_name)
#             })
#
#         print(f"✅ Scanning Complete. Found: {len(self.valid_data)} valid pairs.")
#         if len(self.valid_data) == 0:
#             raise RuntimeError("❌ 没有匹配到数据！请检查 inputs 文件夹路径结构。")
#
#     def __len__(self):
#         return len(self.valid_data)
#
#     def __getitem__(self, idx):
#         data_info = self.valid_data[idx]
#
#         img_obj = Image.open(data_info['img_path']).convert('RGB')
#         image = np.array(img_obj)
#
#         mask_obj = Image.open(data_info['mask_path']).convert('L')
#         mask = np.array(mask_obj)
#
#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image, mask = augmented['image'], augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask.long(), data_info['id']
#
#
# def train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.train()
#     pbar = tqdm(total=len(train_loader), desc=f"Ep {epoch + 1} Train", leave=True)
#     accum_steps = config.get('accumulation_steps', 1)
#
#     current_weights = [1.0, 0.4, 0.3, 0.2, 0.1]
#     if epoch > config['epochs'] * 0.8: current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
#
#     optimizer.zero_grad()
#
#     for i, (input, target, _) in enumerate(train_loader):
#         input = input.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         with autocast(enabled=config['amp']):
#             outputs = model(input)
#
#             if config['deep_supervision'] and isinstance(outputs, (list, tuple)):
#                 loss = 0
#                 for idx, o in enumerate(outputs):
#                     if idx >= len(current_weights) or current_weights[idx] == 0: continue
#                     if o.shape[2:] != target.shape[1:]:
#                         o = F.interpolate(o, size=target.shape[1:], mode='bilinear', align_corners=False)
#                     loss += current_weights[idx] * criterion(o, target)
#                 final_output = outputs[0]
#             else:
#                 loss = criterion(outputs, target)
#                 final_output = outputs
#
#             loss = loss / accum_steps
#
#         if torch.isnan(loss):
#             print(f"⚠️ Warning: Loss is NaN at step {i}. Skipping batch.")
#             optimizer.zero_grad()
#             continue
#
#         if config['amp'] and scaler is not None:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()
#
#         if (i + 1) % accum_steps == 0:
#             if config['amp'] and scaler is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#             optimizer.zero_grad()
#             ema_model.update_parameters(model)
#             scheduler.step()
#
#         with torch.no_grad():
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output_metric = F.interpolate(final_output, size=target.shape[1:], mode='bilinear',
#                                                     align_corners=False)
#             else:
#                 final_output_metric = final_output
#             iou, _, _, _, _, _ = indicators(final_output_metric, target, compute_hd95=False)
#
#         avg_meters['loss'].update(loss.item() * accum_steps, input.size(0))
#         avg_meters['iou'].update(iou, input.size(0))
#         pbar.set_postfix({'L': f"{avg_meters['loss'].avg:.4f}", 'IoU': f"{avg_meters['iou'].avg:.4f}"})
#         pbar.update(1)
#
#     pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def validate(config, val_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.eval()
#     torch.cuda.empty_cache()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(val_loader), desc="Validating", leave=False)
#         for input, target, _ in val_loader:
#             input = input.to(device, non_blocking=True)
#             target = target.to(device, non_blocking=True)
#
#             with autocast(enabled=config['amp']):
#                 outputs = model(input)
#
#             if isinstance(outputs, (list, tuple)):
#                 final_output = outputs[0]
#             else:
#                 final_output = outputs
#
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output = F.interpolate(final_output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             # 为了计算稳定，把输出转回 fp32
#             final_output = final_output.float()
#
#             loss = criterion(final_output, target)
#             iou, _, _, _, _, _ = indicators(final_output, target, compute_hd95=False)
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#             pbar.update(1)
#         pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def main():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#
#     print(f"✅ Device: {device}")
#     print(
#         f"🚀 Mode: RTX 5060 AMP Fast (Workers: {config['num_workers']}, Res: {config['input_w']}, AMP: {config['amp']})")
#
#     base_dir = config['data_dir']
#     dataset_name = config['dataset']
#     images_dir = os.path.join(base_dir, dataset_name, 'images')
#     masks_dir = os.path.join(base_dir, dataset_name, 'masks')
#
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [os.path.splitext(f)[0] for f in all_files_in_dir if
#                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     print(f"   -> 扫描到 {len(valid_ids)} 张图片ID")
#     train_ids, val_ids = train_test_split(valid_ids, test_size=0.2, random_state=config['dataseed'])
#
#     os.makedirs(config['output_dir'], exist_ok=True)
#     save_path = os.path.join(config['output_dir'], config['name'])
#     os.makedirs(save_path, exist_ok=True)
#     with open(os.path.join(save_path, 'config.yml'), 'w') as f:
#         yaml.dump(config, f)
#
#     model = archs.__dict__[config['arch']](
#         num_classes=config['num_classes'],
#         input_channels=config['input_channels'],
#         deep_supervision=config['deep_supervision'],
#         embed_dims=config['input_list']
#     ).to(device)
#
#     decay = 0.999
#     ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
#         decay * averaged_model_parameter + (1.0 - decay) * model_parameter
#     ema_model = AveragedModel(model, avg_fn=ema_avg)
#
#     try:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes'], label_smoothing=0.1).to(device)
#     except:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes']).to(device)
#
#     optimizer = optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])
#     scaler = torch.cuda.amp.GradScaler() if config['amp'] else None
#
#     print("\n=> Preparing Data...")
#
#     train_tf = Compose([
#         Resize(config['input_h'], config['input_w']),
#         HorizontalFlip(p=config.get('HorizontalFlip', 0.5)),
#         VerticalFlip(p=0.5),
#         Normalize(), ToTensorV2()
#     ])
#     val_tf = Compose([Resize(config['input_h'], config['input_w']), Normalize(), ToTensorV2()])
#
#     train_ds = LazyDataset(train_ids, images_dir, masks_dir, config['num_classes'], train_tf)
#     val_ds = LazyDataset(val_ids, images_dir, masks_dir, config['num_classes'], val_tf)
#
#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=(config['num_workers'] > 0)
#     )
#
#     val_loader = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         persistent_workers=(config['num_workers'] > 0)
#     )
#
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=config['max_lr'], epochs=config['epochs'],
#         steps_per_epoch=len(train_loader) // config['accumulation_steps'],
#         pct_start=0.3, div_factor=25, final_div_factor=10000, anneal_strategy='cos'
#     )
#
#     best_iou = 0
#     print(f"\n🚀 Start Training... (Val every 10 epochs)")
#
#     for epoch in range(config['epochs']):
#         print(f'\nEpoch [{epoch + 1}/{config["epochs"]}]')
#         start = time.time()
#
#         train_log = train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler)
#
#         val_str = "Skipped"
#
#         if (epoch + 1) % 10 == 0 or epoch > config['epochs'] - 20:
#             val_log = validate(config, val_loader, ema_model, criterion)
#             val_iou = val_log['iou']
#             val_str = f"{val_iou:.4f}"
#             if val_iou > best_iou:
#                 best_iou = val_iou
#                 torch.save(ema_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
#                 print(f"⭐ New Best IoU: {best_iou:.4f}")
#
#         curr_lr = optimizer.param_groups[0]['lr']
#         print(
#             f"   Time: {time.time() - start:.1f}s | LR: {curr_lr:.2e} | Train IoU: {train_log['iou']:.4f} | Val IoU: {val_str}")
#
#
# if __name__ == '__main__':
#     main()







#我论文标准的train函数，但是加了在每一轮都会记录iou的数据
# import os
# import sys
#
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"
# os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import cv2
#
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
#
# import random
# import numpy as np
# import pandas as pd  # 🟢 【修改1】：引入 pandas 用于保存 CSV 记录
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import yaml
# import time
# import warnings
# from glob import glob
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from torch.cuda.amp import GradScaler, autocast
# from torch.optim.swa_utils import AveragedModel
# from albumentations import (
#     Compose, Resize, HorizontalFlip, VerticalFlip,
#     Normalize, ShiftScaleRotate, RandomBrightnessContrast, HueSaturationValue
# )
# from albumentations.pytorch import ToTensorV2
# from PIL import Image
#
# if hasattr(torch, '_dynamo'):
#     torch._dynamo.config.disable = True
#     torch._dynamo.config.suppress_errors = True
#
# import archs
# import losses
# from metrics import indicators
# from utils import AverageMeter
#
# warnings.filterwarnings("ignore")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # ================= 配置区域 =================
# config = {
#     'name': 'UNet_Classic',
#     'epochs': 300,
#
#     # 🟢 【提速与显存优化 1】：分辨率降到经典的 256x256
#     'input_h': 256,
#     'input_w': 256,
#
#     'batch_size': 4,
#     'accumulation_steps': 2,
#     'num_workers': 4,
#     'dataseed': 3407,
#
#     'arch': 'UNet_Classic',
#     'deep_supervision': True,
#     'input_channels': 3,
#     'num_classes': 8,
#
#     # 🟢 【提速与显存优化 2】：通道数减半，完美抵消 stride=1 带来的空间翻倍
#     'input_list': [32, 64, 128],
#
#     'loss': 'DiceCELoss',
#     'optimizer': 'AdamW',
#
#     # 🟢 【大提速】：重新开启 AMP 混合精度！因为在 arch 里加了安全层，现在不会 NaN 了
#     'amp': True,
#
#     'max_lr': 1e-3,
#     'weight_decay': 1e-2,
#
#     'dataset': 'custom',
#     'data_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs',
#     'output_dir': 'outputs',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#     'HorizontalFlip': 0.5,
#     'multi_scale_training': False
# }
#
#
# class LazyDataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None):
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.num_classes = num_classes
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir) if
#                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir) if
#                                 f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
#
#         print(f"⚡ Verifying image paths in: {img_dir}")
#         self.valid_data = []
#
#         for img_id in tqdm(img_ids, desc="Scanning Paths"):
#             img_name = None
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (img_id + ext).lower()
#                 if test_name in self.real_img_files:
#                     img_name = self.real_img_files[test_name]
#                     break
#             if img_name is None: continue
#
#             mask_name = None
#             target_mask_base = img_id + self.MASK_SUFFIX
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 test_name = (target_mask_base + ext).lower()
#                 if test_name in self.real_mask_files:
#                     mask_name = self.real_mask_files[test_name]
#                     break
#             if mask_name is None:
#                 for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                     test_name = (img_id + ext).lower()
#                     if test_name in self.real_mask_files:
#                         mask_name = self.real_mask_files[test_name]
#                         break
#             if mask_name is None: continue
#
#             self.valid_data.append({
#                 'id': img_id,
#                 'img_path': os.path.join(self.img_dir, img_name),
#                 'mask_path': os.path.join(self.mask_dir, mask_name)
#             })
#
#         print(f"✅ Scanning Complete. Found: {len(self.valid_data)} valid pairs.")
#         if len(self.valid_data) == 0:
#             raise RuntimeError("❌ 没有匹配到数据！请检查 inputs 文件夹路径结构。")
#
#     def __len__(self):
#         return len(self.valid_data)
#
#     def __getitem__(self, idx):
#         data_info = self.valid_data[idx]
#
#         img_obj = Image.open(data_info['img_path']).convert('RGB')
#         image = np.array(img_obj)
#
#         mask_obj = Image.open(data_info['mask_path']).convert('L')
#         mask = np.array(mask_obj)
#
#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image, mask = augmented['image'], augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask.long(), data_info['id']
#
#
# def train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.train()
#     pbar = tqdm(total=len(train_loader), desc=f"Ep {epoch + 1} Train", leave=True)
#     accum_steps = config.get('accumulation_steps', 1)
#
#     current_weights = [1.0, 0.2, 0.3, 0.3, 0.5]
#     if epoch > config['epochs'] * 0.8: current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
#
#     optimizer.zero_grad()
#
#     for i, (input, target, _) in enumerate(train_loader):
#         input = input.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         with autocast(enabled=config['amp']):
#             outputs = model(input)
#
#             if config['deep_supervision'] and isinstance(outputs, (list, tuple)):
#                 loss = 0
#                 for idx, o in enumerate(outputs):
#                     if idx >= len(current_weights) or current_weights[idx] == 0: continue
#                     if o.shape[2:] != target.shape[1:]:
#                         o = F.interpolate(o, size=target.shape[1:], mode='bilinear', align_corners=False)
#                     loss += current_weights[idx] * criterion(o, target)
#                 final_output = outputs[0]
#             else:
#                 loss = criterion(outputs, target)
#                 final_output = outputs
#
#             loss = loss / accum_steps
#
#         if torch.isnan(loss):
#             print(f"⚠️ Warning: Loss is NaN at step {i}. Skipping batch.")
#             optimizer.zero_grad()
#             continue
#
#         if config['amp'] and scaler is not None:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()
#
#         if (i + 1) % accum_steps == 0:
#             if config['amp'] and scaler is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#             optimizer.zero_grad()
#             ema_model.update_parameters(model)
#             scheduler.step()
#
#         with torch.no_grad():
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output_metric = F.interpolate(final_output, size=target.shape[1:], mode='bilinear',
#                                                     align_corners=False)
#             else:
#                 final_output_metric = final_output
#             iou, _, _, _, _, _ = indicators(final_output_metric, target, compute_hd95=False)
#
#         avg_meters['loss'].update(loss.item() * accum_steps, input.size(0))
#         avg_meters['iou'].update(iou, input.size(0))
#         pbar.set_postfix({'L': f"{avg_meters['loss'].avg:.4f}", 'IoU': f"{avg_meters['iou'].avg:.4f}"})
#         pbar.update(1)
#
#     pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def validate(config, val_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
#     model.eval()
#     torch.cuda.empty_cache()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(val_loader), desc="Validating", leave=False)
#         for input, target, _ in val_loader:
#             input = input.to(device, non_blocking=True)
#             target = target.to(device, non_blocking=True)
#
#             with autocast(enabled=config['amp']):
#                 outputs = model(input)
#
#             if isinstance(outputs, (list, tuple)):
#                 final_output = outputs[0]
#             else:
#                 final_output = outputs
#
#             if final_output.shape[2:] != target.shape[1:]:
#                 final_output = F.interpolate(final_output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             # 为了计算稳定，把输出转回 fp32
#             final_output = final_output.float()
#
#             loss = criterion(final_output, target)
#             iou, _, _, _, _, _ = indicators(final_output, target, compute_hd95=False)
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#             pbar.update(1)
#         pbar.close()
#     return {k: v.avg for k, v in avg_meters.items()}
#
#
# def main():
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#
#     print(f"✅ Device: {device}")
#     print(
#         f"🚀 Mode: RTX 5060 AMP Fast (Workers: {config['num_workers']}, Res: {config['input_w']}, AMP: {config['amp']})")
#
#     base_dir = config['data_dir']
#     dataset_name = config['dataset']
#     images_dir = os.path.join(base_dir, dataset_name, 'images')
#     masks_dir = os.path.join(base_dir, dataset_name, 'masks')
#
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [os.path.splitext(f)[0] for f in all_files_in_dir if
#                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#     print(f"   -> 扫描到 {len(valid_ids)} 张图片ID")
#
#     # 🟢 【修改2】：将 test_size=0.2 改为了 0.3，实现 7:3 划分
#     train_ids, val_ids = train_test_split(valid_ids, test_size=0.3, random_state=config['dataseed'])
#     print(f"   -> 数据划分: 训练集 {len(train_ids)} 张, 测试集 {len(val_ids)} 张")
#
#     os.makedirs(config['output_dir'], exist_ok=True)
#     save_path = os.path.join(config['output_dir'], config['name'])
#     os.makedirs(save_path, exist_ok=True)
#     with open(os.path.join(save_path, 'config.yml'), 'w') as f:
#         yaml.dump(config, f)
#
#     model = archs.__dict__[config['arch']](
#         num_classes=config['num_classes'],
#         input_channels=config['input_channels'],
#         deep_supervision=config['deep_supervision'],
#         embed_dims=config['input_list']
#     ).to(device)
#
#     decay = 0.999
#     ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
#         decay * averaged_model_parameter + (1.0 - decay) * model_parameter
#     ema_model = AveragedModel(model, avg_fn=ema_avg)
#
#     try:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes'], label_smoothing=0.1).to(device)
#     except:
#         criterion = losses.__dict__[config['loss']](num_classes=config['num_classes']).to(device)
#
#     optimizer = optim.AdamW(model.parameters(), lr=config['max_lr'], weight_decay=config['weight_decay'])
#     scaler = torch.cuda.amp.GradScaler() if config['amp'] else None
#
#     print("\n=> Preparing Data...")
#
#     train_tf = Compose([
#         Resize(config['input_h'], config['input_w']),
#         HorizontalFlip(p=config.get('HorizontalFlip', 0.5)),
#         VerticalFlip(p=0.5),
#         Normalize(), ToTensorV2()
#     ])
#     val_tf = Compose([Resize(config['input_h'], config['input_w']), Normalize(), ToTensorV2()])
#
#     train_ds = LazyDataset(train_ids, images_dir, masks_dir, config['num_classes'], train_tf)
#     val_ds = LazyDataset(val_ids, images_dir, masks_dir, config['num_classes'], val_tf)
#
#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=(config['num_workers'] > 0)
#     )
#
#     val_loader = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['num_workers'],
#         pin_memory=True,
#         persistent_workers=(config['num_workers'] > 0)
#     )
#
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer, max_lr=config['max_lr'], epochs=config['epochs'],
#         steps_per_epoch=len(train_loader) // config['accumulation_steps'],
#         pct_start=0.3, div_factor=25, final_div_factor=10000, anneal_strategy='cos'
#     )
#
#     best_iou = 0
#     print(f"\n🚀 Start Training... (Validating every epoch and logging to CSV)")
#
#     # 🟢 【修改3】：初始化记录列表与 CSV 保存路径
#     training_log_list = []
#     csv_log_path = os.path.join(save_path, 'training_log.csv')
#
#     for epoch in range(config['epochs']):
#         print(f'\nEpoch [{epoch + 1}/{config["epochs"]}]')
#         start = time.time()
#
#         train_log = train(config, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler)
#
#         # 🟢 【修改4】：去掉 if 限制，训练多少轮就验证多少轮
#         val_log = validate(config, val_loader, ema_model, criterion)
#         val_iou = val_log['iou']
#         val_str = f"{val_iou:.4f}"
#
#         if val_iou > best_iou:
#             best_iou = val_iou
#             torch.save(ema_model.state_dict(), os.path.join(save_path, 'model_best.pth'))
#             print(f"⭐ New Best IoU: {best_iou:.4f} (Model Saved!)")
#
#         curr_lr = optimizer.param_groups[0]['lr']
#         print(
#             f"   Time: {time.time() - start:.1f}s | LR: {curr_lr:.2e} | Train IoU: {train_log['iou']:.4f} | Val IoU: {val_str}")
#
#         # 🟢 【修改5】：将每一轮的指标存入列表并写入 CSV，保存在权重同一目录下
#         training_log_list.append({
#             'Epoch': epoch + 1,
#             'Train_Loss': train_log['loss'],
#             'Train_IoU': train_log['iou'],
#             'Val_Loss': val_log['loss'],
#             'Val_IoU': val_log['iou'],
#             'Learning_Rate': curr_lr
#         })
#         pd.DataFrame(training_log_list).to_csv(csv_log_path, index=False)
#
#
# if __name__ == '__main__':
#     main()
#








# train_busi_unet_binary.py
# 作用：训练 BUSI 二值病灶分割模型
# 数据结构：
# D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\inputs\busi
# ├── images
# │   ├── malignant (1).png / .jpg / .bmp ...
# └── masks
#     ├── malignant (1)_mask.png / .jpg / .bmp ...
#
# import os
# import random
# import time
# import warnings
# from pathlib import Path
#
# os.environ["TORCH_COMPILE_DISABLE"] = "1"
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
#
# import yaml
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
# from torch.optim.swa_utils import AveragedModel
#
# from albumentations import (
#     Compose, Resize, HorizontalFlip, VerticalFlip,
#     ShiftScaleRotate, RandomBrightnessContrast, Normalize
# )
# from albumentations.pytorch import ToTensorV2
#
# import archs
#
# warnings.filterwarnings("ignore")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ========================= 1. 配置区域，只需要主要改这里 =========================
# config = {
#     # 输出文件夹名称
#     "name": "UNet_cvc_binary",
#
#     # 训练轮数：BUSI样本少，可以先跑 100 看效果，再加到 200~300
#     "epochs": 200,
#
#     # 输入尺寸
#     "input_h": 256,
#     "input_w": 256,
#
#     "batch_size": 4,
#     "accumulation_steps": 2,
#     "num_workers": 0,          # Windows/Pycharm 下建议先用 0，稳定后可改 2 或 4
#     "dataseed": 3407,
#
#     # 模型参数：必须与你 archs.py 里的模型名称一致
#     "arch": "UNet_Classic",
#     "deep_supervision": True,
#     "input_channels": 3,
#
#     # 二值分割：病灶=1，背景=0，所以输出通道为 1
#     "num_classes": 1,
#
#     # 如果你的 UNet_Classic 支持 embed_dims，就会使用；不支持则自动跳过
#     "input_list": [32, 64, 128],
#
#     "amp": True,
#     "max_lr": 1e-3,
#     "weight_decay": 1e-2,
#
#     # 你的 BUSI 数据集总路径
#     "data_root": r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\inputs\cvc_220",
#
#     # 训练结果保存位置
#     "output_dir": "outputs",
#
#     # 图像与mask命名规则
#     "mask_suffix": "_mask",
#
#     # 训练/验证划分，0.3 表示 70%训练，30%验证
#     "val_ratio": 0.3,
#
#     # 二值化阈值
#     "threshold": 0.5,
# }
#
#
# # ========================= 2. 固定随机种子 =========================
# def seed_everything(seed=3407):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# # ========================= 3. 简单平均器 =========================
# class AverageMeter:
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = float(val)
#         self.sum += float(val) * n
#         self.count += n
#         self.avg = self.sum / max(self.count, 1)
#
#
# # ========================= 4. 二值 Dice + BCE 损失 =========================
# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_weight = bce_weight
#         self.dice_weight = dice_weight
#         self.smooth = smooth
#
#     def forward(self, logits, target):
#         # logits: [B, 1, H, W]
#         # target: [B, 1, H, W]，取值 0 或 1
#         if logits.shape[2:] != target.shape[2:]:
#             logits = F.interpolate(logits, size=target.shape[2:], mode="bilinear", align_corners=False)
#
#         target = target.float()
#         bce_loss = self.bce(logits, target)
#
#         prob = torch.sigmoid(logits)
#         prob = prob.contiguous().view(prob.size(0), -1)
#         target = target.contiguous().view(target.size(0), -1)
#
#         intersection = (prob * target).sum(dim=1)
#         dice = (2.0 * intersection + self.smooth) / (
#             prob.sum(dim=1) + target.sum(dim=1) + self.smooth
#         )
#         dice_loss = 1.0 - dice.mean()
#
#         return self.bce_weight * bce_loss + self.dice_weight * dice_loss
#
#
# # ========================= 5. 二值 IoU / Dice 指标 =========================
# @torch.no_grad()
# def binary_iou_dice(logits, target, threshold=0.5, eps=1e-7):
#     if logits.shape[2:] != target.shape[2:]:
#         logits = F.interpolate(logits, size=target.shape[2:], mode="bilinear", align_corners=False)
#
#     prob = torch.sigmoid(logits)
#     pred = (prob > threshold).float()
#     target = (target > 0.5).float()
#
#     pred = pred.view(pred.size(0), -1)
#     target = target.view(target.size(0), -1)
#
#     intersection = (pred * target).sum(dim=1)
#     union = pred.sum(dim=1) + target.sum(dim=1) - intersection
#
#     iou = ((intersection + eps) / (union + eps)).mean().item()
#     dice = ((2.0 * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)).mean().item()
#     return iou, dice
#
#
# # ========================= 6. BUSI 数据集读取 =========================
# class BUSIBinaryDataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, mask_suffix="_mask", transform=None):
#         self.img_ids = img_ids
#         self.img_dir = Path(img_dir)
#         self.mask_dir = Path(mask_dir)
#         self.mask_suffix = mask_suffix
#         self.transform = transform
#         self.exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
#
#         self.real_img_files = {
#             f.name.lower(): f.name
#             for f in self.img_dir.iterdir()
#             if f.is_file() and f.suffix.lower() in self.exts
#         }
#         self.real_mask_files = {
#             f.name.lower(): f.name
#             for f in self.mask_dir.iterdir()
#             if f.is_file() and f.suffix.lower() in self.exts
#         }
#
#         self.valid_data = []
#         print(f"⚡ 正在检查图像路径: {self.img_dir}")
#         print(f"⚡ 正在检查mask路径: {self.mask_dir}")
#
#         for img_id in tqdm(img_ids, desc="Scanning BUSI pairs"):
#             img_name = self._find_file(self.real_img_files, img_id)
#             if img_name is None:
#                 continue
#
#             # 主要匹配方式：malignant (1) -> malignant (1)_mask
#             mask_name = self._find_file(self.real_mask_files, img_id + self.mask_suffix)
#
#             # 备用匹配方式：如果有些mask和原图同名，也允许匹配
#             if mask_name is None:
#                 mask_name = self._find_file(self.real_mask_files, img_id)
#
#             if mask_name is None:
#                 print(f"⚠️ 未找到对应mask: {img_id}")
#                 continue
#
#             self.valid_data.append({
#                 "id": img_id,
#                 "img_path": str(self.img_dir / img_name),
#                 "mask_path": str(self.mask_dir / mask_name),
#             })
#
#         print(f"✅ 有效图像-mask对数: {len(self.valid_data)}")
#         if len(self.valid_data) == 0:
#             raise RuntimeError("没有匹配到任何 image-mask 对。请检查 images/masks 文件夹和 _mask 命名。")
#
#     def _find_file(self, file_dict, base_name):
#         for ext in self.exts:
#             key = (base_name + ext).lower()
#             if key in file_dict:
#                 return file_dict[key]
#         return None
#
#     def __len__(self):
#         return len(self.valid_data)
#
#     def __getitem__(self, idx):
#         item = self.valid_data[idx]
#
#         image = np.array(Image.open(item["img_path"]).convert("RGB"))
#         mask = np.array(Image.open(item["mask_path"]).convert("L"))
#
#         # 原始mask是黑白图，白色病灶区域可能是255，这里统一转成0/1
#         mask = (mask > 127).astype("float32")
#
#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented["image"]
#             mask = augmented["mask"]
#         else:
#             image = cv2.resize(image, (config["input_w"], config["input_h"]))
#             mask = cv2.resize(mask, (config["input_w"], config["input_h"]), interpolation=cv2.INTER_NEAREST)
#             image = image.astype("float32") / 255.0
#             image = torch.from_numpy(image.transpose(2, 0, 1))
#             mask = torch.from_numpy(mask)
#
#         # ToTensorV2 后 mask 是 [H, W]，这里补成 [1, H, W]
#         if not torch.is_tensor(mask):
#             mask = torch.from_numpy(mask)
#         mask = (mask > 0.5).float().unsqueeze(0)
#
#         return image.float(), mask.float(), item["id"]
#
#
# # ========================= 7. 构建模型 =========================
# def build_model(cfg):
#     model_cls = archs.__dict__[cfg["arch"]]
#
#     # 兼容不同写法的 UNet_Classic
#     try:
#         model = model_cls(
#             num_classes=cfg["num_classes"],
#             input_channels=cfg["input_channels"],
#             deep_supervision=cfg["deep_supervision"],
#             embed_dims=cfg["input_list"],
#         )
#     except TypeError:
#         try:
#             model = model_cls(
#                 num_classes=cfg["num_classes"],
#                 input_channels=cfg["input_channels"],
#                 deep_supervision=cfg["deep_supervision"],
#             )
#         except TypeError:
#             model = model_cls(
#                 in_channels=cfg["input_channels"],
#                 num_classes=cfg["num_classes"],
#             )
#
#     return model.to(device)
#
#
# # ========================= 8. 单轮训练 =========================
# def train_one_epoch(cfg, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
#     avg_loss = AverageMeter()
#     avg_iou = AverageMeter()
#     avg_dice = AverageMeter()
#
#     model.train()
#     accum_steps = cfg.get("accumulation_steps", 1)
#
#     # 深监督权重：第一个输出是主输出，后面是辅助输出
#     current_weights = [1.0, 0.4, 0.3, 0.2, 0.1]
#     if epoch > cfg["epochs"] * 0.8:
#         current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
#
#     optimizer.zero_grad(set_to_none=True)
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", leave=True)
#
#     for i, (images, masks, _) in enumerate(pbar):
#         images = images.to(device, non_blocking=True)
#         masks = masks.to(device, non_blocking=True)
#
#         with autocast(enabled=cfg["amp"]):
#             outputs = model(images)
#
#             if cfg["deep_supervision"] and isinstance(outputs, (list, tuple)):
#                 loss = 0.0
#                 for idx, out in enumerate(outputs):
#                     if idx >= len(current_weights) or current_weights[idx] == 0:
#                         continue
#                     loss = loss + current_weights[idx] * criterion(out, masks)
#                 final_output = outputs[0]
#             else:
#                 loss = criterion(outputs, masks)
#                 final_output = outputs
#
#             loss = loss / accum_steps
#
#         if torch.isnan(loss):
#             print(f"⚠️ 第 {i} 个 batch 出现 NaN，已跳过")
#             optimizer.zero_grad(set_to_none=True)
#             continue
#
#         if cfg["amp"] and scaler is not None:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()
#
#         do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(train_loader))
#         if do_step:
#             if cfg["amp"] and scaler is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#
#             optimizer.zero_grad(set_to_none=True)
#             ema_model.update_parameters(model)
#             scheduler.step()
#
#         iou, dice = binary_iou_dice(final_output.float(), masks, threshold=cfg["threshold"])
#
#         avg_loss.update(loss.item() * accum_steps, images.size(0))
#         avg_iou.update(iou, images.size(0))
#         avg_dice.update(dice, images.size(0))
#
#         pbar.set_postfix({
#             "loss": f"{avg_loss.avg:.4f}",
#             "iou": f"{avg_iou.avg:.4f}",
#             "dice": f"{avg_dice.avg:.4f}",
#         })
#
#     return {
#         "loss": avg_loss.avg,
#         "iou": avg_iou.avg,
#         "dice": avg_dice.avg,
#     }
#
#
# # ========================= 9. 验证 =========================
# @torch.no_grad()
# def validate(cfg, val_loader, model, criterion):
#     avg_loss = AverageMeter()
#     avg_iou = AverageMeter()
#     avg_dice = AverageMeter()
#
#     model.eval()
#     torch.cuda.empty_cache()
#
#     pbar = tqdm(val_loader, desc="Validating", leave=False)
#     for images, masks, _ in pbar:
#         images = images.to(device, non_blocking=True)
#         masks = masks.to(device, non_blocking=True)
#
#         with autocast(enabled=cfg["amp"]):
#             outputs = model(images)
#
#         final_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
#         final_output = final_output.float()
#
#         loss = criterion(final_output, masks)
#         iou, dice = binary_iou_dice(final_output, masks, threshold=cfg["threshold"])
#
#         avg_loss.update(loss.item(), images.size(0))
#         avg_iou.update(iou, images.size(0))
#         avg_dice.update(dice, images.size(0))
#
#     return {
#         "loss": avg_loss.avg,
#         "iou": avg_iou.avg,
#         "dice": avg_dice.avg,
#     }
#
#
# # ========================= 10. 主函数 =========================
# def main():
#     seed_everything(config["dataseed"])
#
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#
#     print(f"✅ Device: {device}")
#
#     data_root = Path(config["data_root"])
#     images_dir = data_root / "images"
#     masks_dir = data_root / "masks"
#
#     if not images_dir.exists():
#         raise FileNotFoundError(f"找不到 images 文件夹: {images_dir}")
#     if not masks_dir.exists():
#         raise FileNotFoundError(f"找不到 masks 文件夹: {masks_dir}")
#
#     valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
#     all_img_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
#     valid_ids = [p.stem for p in all_img_files]
#
#     print(f"📌 扫描到原始图像: {len(valid_ids)} 张")
#
#     train_ids, val_ids = train_test_split(
#         valid_ids,
#         test_size=config["val_ratio"],
#         random_state=config["dataseed"],
#         shuffle=True,
#     )
#
#     print(f"📌 数据划分: 训练集 {len(train_ids)} 张, 验证/测试集 {len(val_ids)} 张")
#
#     save_path = Path(config["output_dir"]) / config["name"]
#     save_path.mkdir(parents=True, exist_ok=True)
#
#     # 保存配置和划分文件，test.py 可以直接读取 val_ids.txt 作为测试集
#     with open(save_path / "config.yml", "w", encoding="utf-8") as f:
#         yaml.safe_dump(config, f, allow_unicode=True)
#
#     with open(save_path / "train_ids.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(train_ids))
#
#     with open(save_path / "val_ids.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(val_ids))
#
#     train_tf = Compose([
#         Resize(config["input_h"], config["input_w"]),
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.2),
#         ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),
#         RandomBrightnessContrast(p=0.3),
#         Normalize(),
#         ToTensorV2(),
#     ])
#
#     val_tf = Compose([
#         Resize(config["input_h"], config["input_w"]),
#         Normalize(),
#         ToTensorV2(),
#     ])
#
#     train_ds = BUSIBinaryDataset(
#         train_ids,
#         images_dir,
#         masks_dir,
#         mask_suffix=config["mask_suffix"],
#         transform=train_tf,
#     )
#     val_ds = BUSIBinaryDataset(
#         val_ids,
#         images_dir,
#         masks_dir,
#         mask_suffix=config["mask_suffix"],
#         transform=val_tf,
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         num_workers=config["num_workers"],
#         pin_memory=True,
#         drop_last=False,
#         persistent_workers=(config["num_workers"] > 0),
#     )
#
#     val_loader = torch.utils.data.DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=config["num_workers"],
#         pin_memory=True,
#         persistent_workers=(config["num_workers"] > 0),
#     )
#
#     model = build_model(config)
#
#     decay = 0.999
#     ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
#         decay * averaged_model_parameter + (1.0 - decay) * model_parameter
#     )
#     ema_model = AveragedModel(model, avg_fn=ema_avg)
#
#     criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=config["max_lr"], weight_decay=config["weight_decay"])
#     scaler = GradScaler(enabled=config["amp"])
#
#     steps_per_epoch = max(1, int(np.ceil(len(train_loader) / config["accumulation_steps"])))
#     total_steps = config["epochs"] * steps_per_epoch
#
#     scheduler = optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr=config["max_lr"],
#         total_steps=total_steps,
#         pct_start=0.3,
#         div_factor=25,
#         final_div_factor=10000,
#         anneal_strategy="cos",
#     )
#
#     best_iou = -1.0
#     best_dice = -1.0
#     log_list = []
#     csv_log_path = save_path / "training_log.csv"
#
#     print("\n🚀 开始训练 BUSI 二值病灶分割 UNet...\n")
#
#     for epoch in range(config["epochs"]):
#         start_time = time.time()
#
#         train_log = train_one_epoch(
#             config, train_loader, model, ema_model, criterion,
#             optimizer, scaler, epoch, scheduler
#         )
#
#         # 使用 EMA 模型验证，一般比原始模型更稳
#         val_log = validate(config, val_loader, ema_model, criterion)
#
#         curr_lr = optimizer.param_groups[0]["lr"]
#
#         # 保存最优权重：直接保存 ema_model.module.state_dict()
#         # 这样 test.py 加载最简单，不会出现 module. 或 n_averaged 问题
#         if val_log["iou"] > best_iou:
#             best_iou = val_log["iou"]
#             best_dice = val_log["dice"]
#
#             torch.save({
#                 "model": ema_model.module.state_dict(),
#                 "epoch": epoch + 1,
#                 "best_iou": best_iou,
#                 "best_dice": best_dice,
#                 "config": config,
#             }, save_path / "model_best.pth")
#
#             print(f"⭐ 保存最优模型: IoU={best_iou:.4f}, Dice={best_dice:.4f}")
#
#         # 每轮也保存最后模型，防止训练中断
#         torch.save({
#             "model": ema_model.module.state_dict(),
#             "epoch": epoch + 1,
#             "best_iou": best_iou,
#             "best_dice": best_dice,
#             "config": config,
#         }, save_path / "model_last.pth")
#
#         elapsed = time.time() - start_time
#
#         print(
#             f"Epoch [{epoch + 1:03d}/{config['epochs']}] "
#             f"Time: {elapsed:.1f}s | LR: {curr_lr:.2e} | "
#             f"Train Loss: {train_log['loss']:.4f} | Train IoU: {train_log['iou']:.4f} | Train Dice: {train_log['dice']:.4f} | "
#             f"Val Loss: {val_log['loss']:.4f} | Val IoU: {val_log['iou']:.4f} | Val Dice: {val_log['dice']:.4f}"
#         )
#
#         log_list.append({
#             "Epoch": epoch + 1,
#             "Train_Loss": train_log["loss"],
#             "Train_IoU": train_log["iou"],
#             "Train_Dice": train_log["dice"],
#             "Val_Loss": val_log["loss"],
#             "Val_IoU": val_log["iou"],
#             "Val_Dice": val_log["dice"],
#             "Learning_Rate": curr_lr,
#         })
#         pd.DataFrame(log_list).to_csv(csv_log_path, index=False, encoding="utf-8-sig")
#
#     print("\n✅ 训练完成")
#     print(f"📁 最优权重: {save_path / 'model_best.pth'}")
#     print(f"📁 训练日志: {csv_log_path}")
#     print(f"📁 验证集ID: {save_path / 'val_ids.txt'}")
#
#
# if __name__ == "__main__":
#     main()










# train_cvc_unet_binary_safe.py
# 训练 CVC 二值分割 UNet。重点修复：不用 PIL 读取图片，改用 cv2.imdecode；
# 遇到坏图/空图/无法识别图片会自动跳过，并记录到 skipped_files.csv。

import os
import random
import time
import warnings
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel

from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip,
    ShiftScaleRotate, RandomBrightnessContrast, Normalize
)
from albumentations.pytorch import ToTensorV2

import archs

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 1. 配置区域 =========================
config = {
    "name": "UNet_cvc_binary",
    "epochs": 200,

    "input_h": 256,
    "input_w": 256,

    "batch_size": 4,
    "accumulation_steps": 2,
    "num_workers": 0,
    "dataseed": 3407,

    "arch": "UNet_Classic",
    "deep_supervision": True,
    "input_channels": 3,
    "num_classes": 1,
    "input_list": [32, 64, 128],

    "amp": True,
    "max_lr": 1e-3,
    "weight_decay": 1e-2,

    # CVC 训练集路径
    "data_root": r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\inputs\cvc_220",

    "output_dir": "outputs",

    # CVC 通常是 images\40.png 对应 masks\40.png，所以这里用空字符串
    # 程序也会自动兼容 40_mask.png、40_mask_1.png
    "mask_suffix": "",

    "val_ratio": 0.3,
    "threshold": 0.5,

    # True 表示遇到坏图自动跳过，不让训练崩溃
    "skip_bad_files": True,
}


# ========================= 2. 工具函数 =========================
def seed_everything(seed=3407):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def safe_read_rgb_image(img_path):
    """安全读取 RGB 图像。读取失败返回 None。"""
    img_path = Path(img_path)

    if not img_path.exists() or img_path.stat().st_size == 0:
        return None

    try:
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return None

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    except Exception:
        return None


def safe_read_gray_image(mask_path):
    """安全读取灰度 mask。读取失败返回 None。"""
    mask_path = Path(mask_path)

    if not mask_path.exists() or mask_path.stat().st_size == 0:
        return None

    try:
        data = np.fromfile(str(mask_path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return None

        return mask

    except Exception:
        return None


# ========================= 3. 损失函数与指标 =========================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, target):
        if logits.shape[2:] != target.shape[2:]:
            logits = F.interpolate(logits, size=target.shape[2:], mode="bilinear", align_corners=False)

        target = target.float()
        bce_loss = self.bce(logits, target)

        prob = torch.sigmoid(logits)
        prob = prob.contiguous().view(prob.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (prob * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            prob.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def binary_iou_dice(logits, target, threshold=0.5, eps=1e-7):
    if logits.shape[2:] != target.shape[2:]:
        logits = F.interpolate(logits, size=target.shape[2:], mode="bilinear", align_corners=False)

    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    target = (target > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2.0 * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)).mean().item()
    return iou, dice


# ========================= 4. 数据集 =========================
class CVCBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, mask_suffix="", transform=None, skip_bad_files=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.skip_bad_files = skip_bad_files
        self.exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        self.real_img_files = {
            f.stem.lower(): f.name
            for f in self.img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.exts
        }
        self.real_mask_files = {
            f.stem.lower(): f.name
            for f in self.mask_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.exts
        }

        self.valid_data = []
        self.skipped_data = []

        print(f"⚡ 正在检查图像路径: {self.img_dir}")
        print(f"⚡ 正在检查mask路径: {self.mask_dir}")

        for img_id in tqdm(img_ids, desc="Scanning CVC pairs"):
            img_name = self.real_img_files.get(str(img_id).lower(), None)

            if img_name is None:
                self.skipped_data.append({"id": img_id, "reason": "找不到图像文件"})
                continue

            mask_name = self._find_mask_name(str(img_id))

            if mask_name is None:
                self.skipped_data.append({
                    "id": img_id,
                    "image_path": str(self.img_dir / img_name),
                    "reason": "找不到对应mask"
                })
                continue

            img_path = self.img_dir / img_name
            mask_path = self.mask_dir / mask_name

            if self.skip_bad_files:
                if safe_read_rgb_image(img_path) is None:
                    self.skipped_data.append({
                        "id": img_id,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "reason": "图像无法识别或文件损坏"
                    })
                    continue

                if safe_read_gray_image(mask_path) is None:
                    self.skipped_data.append({
                        "id": img_id,
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "reason": "mask无法识别或文件损坏"
                    })
                    continue

            self.valid_data.append({
                "id": str(img_id),
                "img_path": str(img_path),
                "mask_path": str(mask_path),
            })

        print(f"✅ 有效图像-mask对数: {len(self.valid_data)}")
        if len(self.skipped_data) > 0:
            print(f"⚠️ 跳过异常文件: {len(self.skipped_data)} 个")

        if len(self.valid_data) == 0:
            raise RuntimeError("没有匹配到任何可用 image-mask 对。请检查 images/masks 文件夹、命名和图片是否损坏。")

    def _find_mask_name(self, img_id):
        candidates = []

        if self.mask_suffix is not None:
            candidates.append(img_id + self.mask_suffix)

        candidates.extend([img_id, img_id + "_mask", img_id + "_mask_1"])

        unique_candidates = []
        for c in candidates:
            if c not in unique_candidates:
                unique_candidates.append(c)

        for c in unique_candidates:
            name = self.real_mask_files.get(c.lower(), None)
            if name is not None:
                return name

        return None

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]

        image = safe_read_rgb_image(item["img_path"])
        mask = safe_read_gray_image(item["mask_path"])

        if image is None:
            raise RuntimeError(f"读取图像失败: {item['img_path']}")
        if mask is None:
            raise RuntimeError(f"读取mask失败: {item['mask_path']}")

        mask = (mask > 127).astype("float32")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (config["input_w"], config["input_h"]))
            mask = cv2.resize(mask, (config["input_w"], config["input_h"]), interpolation=cv2.INTER_NEAREST)
            image = image.astype("float32") / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask)

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)

        mask = (mask > 0.5).float().unsqueeze(0)

        return image.float(), mask.float(), item["id"]


# ========================= 5. 模型 =========================
def build_model(cfg):
    model_cls = archs.__dict__[cfg["arch"]]

    try:
        model = model_cls(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            deep_supervision=cfg["deep_supervision"],
            embed_dims=cfg["input_list"],
        )
    except TypeError:
        try:
            model = model_cls(
                num_classes=cfg["num_classes"],
                input_channels=cfg["input_channels"],
                deep_supervision=cfg["deep_supervision"],
            )
        except TypeError:
            model = model_cls(
                in_channels=cfg["input_channels"],
                num_classes=cfg["num_classes"],
            )

    return model.to(device)


# ========================= 6. 训练与验证 =========================
def train_one_epoch(cfg, train_loader, model, ema_model, criterion, optimizer, scaler, epoch, scheduler):
    avg_loss = AverageMeter()
    avg_iou = AverageMeter()
    avg_dice = AverageMeter()

    model.train()
    accum_steps = cfg.get("accumulation_steps", 1)

    current_weights = [1.0, 0.4, 0.3, 0.2, 0.1]
    if epoch > cfg["epochs"] * 0.8:
        current_weights = [1.0, 0.0, 0.0, 0.0, 0.0]

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", leave=True)

    for i, (images, masks, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=cfg["amp"]):
            outputs = model(images)

            if cfg["deep_supervision"] and isinstance(outputs, (list, tuple)):
                loss = 0.0
                for idx, out in enumerate(outputs):
                    if idx >= len(current_weights) or current_weights[idx] == 0:
                        continue
                    loss = loss + current_weights[idx] * criterion(out, masks)
                final_output = outputs[0]
            else:
                loss = criterion(outputs, masks)
                final_output = outputs

            loss = loss / accum_steps

        if torch.isnan(loss):
            print(f"⚠️ 第 {i} 个 batch 出现 NaN，已跳过")
            optimizer.zero_grad(set_to_none=True)
            continue

        if cfg["amp"] and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(train_loader))
        if do_step:
            if cfg["amp"] and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            ema_model.update_parameters(model)
            scheduler.step()

        iou, dice = binary_iou_dice(final_output.float(), masks, threshold=cfg["threshold"])

        avg_loss.update(loss.item() * accum_steps, images.size(0))
        avg_iou.update(iou, images.size(0))
        avg_dice.update(dice, images.size(0))

        pbar.set_postfix({
            "loss": f"{avg_loss.avg:.4f}",
            "iou": f"{avg_iou.avg:.4f}",
            "dice": f"{avg_dice.avg:.4f}",
        })

    return {"loss": avg_loss.avg, "iou": avg_iou.avg, "dice": avg_dice.avg}


@torch.no_grad()
def validate(cfg, val_loader, model, criterion):
    avg_loss = AverageMeter()
    avg_iou = AverageMeter()
    avg_dice = AverageMeter()

    model.eval()
    torch.cuda.empty_cache()

    pbar = tqdm(val_loader, desc="Validating", leave=False)

    for images, masks, _ in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=cfg["amp"]):
            outputs = model(images)

        final_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        final_output = final_output.float()

        loss = criterion(final_output, masks)
        iou, dice = binary_iou_dice(final_output, masks, threshold=cfg["threshold"])

        avg_loss.update(loss.item(), images.size(0))
        avg_iou.update(iou, images.size(0))
        avg_dice.update(dice, images.size(0))

    return {"loss": avg_loss.avg, "iou": avg_iou.avg, "dice": avg_dice.avg}


# ========================= 7. 主函数 =========================
def main():
    seed_everything(config["dataseed"])

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f"✅ Device: {device}")

    data_root = Path(config["data_root"])
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 images 文件夹: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"找不到 masks 文件夹: {masks_dir}")

    valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    all_img_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    valid_ids = [p.stem for p in all_img_files]

    print(f"📌 扫描到原始图像: {len(valid_ids)} 张")

    train_ids, val_ids = train_test_split(
        valid_ids,
        test_size=config["val_ratio"],
        random_state=config["dataseed"],
        shuffle=True,
    )

    print(f"📌 初始划分: 训练集 {len(train_ids)} 张, 验证/测试集 {len(val_ids)} 张")

    save_path = Path(config["output_dir"]) / config["name"]
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    train_tf = Compose([
        Resize(config["input_h"], config["input_w"]),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(p=0.3),
        Normalize(),
        ToTensorV2(),
    ])

    val_tf = Compose([
        Resize(config["input_h"], config["input_w"]),
        Normalize(),
        ToTensorV2(),
    ])

    train_ds = CVCBinaryDataset(
        train_ids,
        images_dir,
        masks_dir,
        mask_suffix=config["mask_suffix"],
        transform=train_tf,
        skip_bad_files=config["skip_bad_files"],
    )

    val_ds = CVCBinaryDataset(
        val_ids,
        images_dir,
        masks_dir,
        mask_suffix=config["mask_suffix"],
        transform=val_tf,
        skip_bad_files=config["skip_bad_files"],
    )

    final_train_ids = [d["id"] for d in train_ds.valid_data]
    final_val_ids = [d["id"] for d in val_ds.valid_data]

    with open(save_path / "train_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(final_train_ids))

    with open(save_path / "val_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(final_val_ids))

    skipped_all = train_ds.skipped_data + val_ds.skipped_data
    if len(skipped_all) > 0:
        skipped_csv_path = save_path / "skipped_files.csv"
        pd.DataFrame(skipped_all).to_csv(skipped_csv_path, index=False, encoding="utf-8-sig")
        print(f"⚠️ 已保存异常文件记录: {skipped_csv_path}")

    if len(train_ds) == 0:
        raise RuntimeError("训练集有效样本数为 0，无法训练。")
    if len(val_ds) == 0:
        raise RuntimeError("验证集有效样本数为 0，无法验证。")

    print(f"📌 最终有效训练集: {len(train_ds)} 张")
    print(f"📌 最终有效验证集: {len(val_ds)} 张")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=(config["num_workers"] > 0),
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=(config["num_workers"] > 0),
    )

    model = build_model(config)

    decay = 0.999
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (
        decay * averaged_model_parameter + (1.0 - decay) * model_parameter
    )
    ema_model = AveragedModel(model, avg_fn=ema_avg)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["max_lr"], weight_decay=config["weight_decay"])
    scaler = GradScaler(enabled=config["amp"])

    steps_per_epoch = max(1, int(np.ceil(len(train_loader) / config["accumulation_steps"])))
    total_steps = config["epochs"] * steps_per_epoch

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000,
        anneal_strategy="cos",
    )

    best_iou = -1.0
    best_dice = -1.0
    log_list = []
    csv_log_path = save_path / "training_log.csv"

    print("\n🚀 开始训练 CVC 二值分割 UNet...\n")

    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_log = train_one_epoch(
            config, train_loader, model, ema_model, criterion,
            optimizer, scaler, epoch, scheduler
        )

        val_log = validate(config, val_loader, ema_model, criterion)
        curr_lr = optimizer.param_groups[0]["lr"]

        if val_log["iou"] > best_iou:
            best_iou = val_log["iou"]
            best_dice = val_log["dice"]

            torch.save({
                "model": ema_model.module.state_dict(),
                "epoch": epoch + 1,
                "best_iou": best_iou,
                "best_dice": best_dice,
                "config": config,
            }, save_path / "model_best.pth")

            print(f"⭐ 保存最优模型: IoU={best_iou:.4f}, Dice={best_dice:.4f}")

        torch.save({
            "model": ema_model.module.state_dict(),
            "epoch": epoch + 1,
            "best_iou": best_iou,
            "best_dice": best_dice,
            "config": config,
        }, save_path / "model_last.pth")

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch + 1:03d}/{config['epochs']}] "
            f"Time: {elapsed:.1f}s | LR: {curr_lr:.2e} | "
            f"Train Loss: {train_log['loss']:.4f} | Train IoU: {train_log['iou']:.4f} | Train Dice: {train_log['dice']:.4f} | "
            f"Val Loss: {val_log['loss']:.4f} | Val IoU: {val_log['iou']:.4f} | Val Dice: {val_log['dice']:.4f}"
        )

        log_list.append({
            "Epoch": epoch + 1,
            "Train_Loss": train_log["loss"],
            "Train_IoU": train_log["iou"],
            "Train_Dice": train_log["dice"],
            "Val_Loss": val_log["loss"],
            "Val_IoU": val_log["iou"],
            "Val_Dice": val_log["dice"],
            "Learning_Rate": curr_lr,
        })
        pd.DataFrame(log_list).to_csv(csv_log_path, index=False, encoding="utf-8-sig")

    print("\n✅ 训练完成")
    print(f"📁 最优权重: {save_path / 'model_best.pth'}")
    print(f"📁 最后权重: {save_path / 'model_last.pth'}")
    print(f"📁 训练日志: {csv_log_path}")
    print(f"📁 验证集ID: {save_path / 'val_ids.txt'}")


if __name__ == "__main__":
    main()
