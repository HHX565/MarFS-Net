# import os
# import sys
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import cv2
# from glob import glob
# from tqdm import tqdm
# from collections import OrderedDict
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
#
# # === 新增：用于确保数据切分与训练时绝对一致 ===
# from sklearn.model_selection import train_test_split
#
# # === 增强库 (与训练保持一致) ===
# from albumentations import Compose, Resize, Normalize
# from albumentations.pytorch import ToTensorV2
#
# # === 本地模块 ===
# import archs
#
# # 尝试导入 metrics，如果不存在则使用内置的简单计算函数
# try:
#     from metrics import indicators
# except ImportError:
#     print("⚠️ 警告: 未找到 metrics.py，将使用内置的基础 IoU 计算函数")
#
#     def indicators(output, target, compute_hd95=False):
#         # 简易版指标计算，防止缺失文件报错
#         pred = torch.argmax(output, dim=1).cpu().numpy()
#         target = target.cpu().numpy()
#         intersection = np.logical_and(target == pred, target > 0).sum()
#         union = np.logical_or(target == pred, target > 0).sum()
#         iou = (intersection + 1e-6) / (union + 1e-6)
#         return iou, 0, 0, 0, 0, 0
#
# # ================= 配置区域 =================
# config = {
#     # 1. 架构名称 (必须对应 archs.py 中的类名)
#     'arch': 'NSFPN_Official',
#
#     # 🟢 【核心修改 1】：必须与训练时的 input_list 严格一致！
#     'embed_dims': [32, 64, 128],
#     'num_classes': 8,
#     'input_channels': 3,
#
#     'deep_supervision': False,
#
#     # 🟢 【核心修改 2】：与训练时的分辨率严格一致！
#     'input_h': 256,
#     'input_w': 256,
#
#     # 4. 路径配置 (⚠️ 请务必替换为你真实的权重路径)
#     'model_path': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\outputs\NSFPN_Official\model_best.pth',
#     'dataset_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs\custom',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#
#     # 5. 【🔥 核心防泄漏配置】必须与 train.py 的随机种子完全一致！
#     'dataseed': 3407,
# }
#
#
# # ================= 测试数据集类 =================
# class TestDataset(Dataset):
#     """
#     复刻 train.py 中的数据读取逻辑，只读取被分配到测试集的 img_ids
#     """
#
#     def __init__(self, img_ids, img_dir, mask_dir, transform=None):
#         self.img_ids = img_ids  # 直接接收切分好的测试集 ID
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         # 建立文件映射缓存
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir)}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir)}
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#
#         # 1. 查找图片
#         img_name = None
#         for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#             key = (img_id + ext).lower()
#             if key in self.real_img_files:
#                 img_name = self.real_img_files[key]
#                 break
#
#         # 2. 查找掩码 (优先匹配带后缀的)
#         mask_name = None
#         target_mask_base = img_id + self.MASK_SUFFIX
#         for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#             key = (target_mask_base + ext).lower()
#             if key in self.real_mask_files:
#                 mask_name = self.real_mask_files[key]
#                 break
#
#         # 回退查找 (匹配不带后缀的)
#         if mask_name is None:
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 key = (img_id + ext).lower()
#                 if key in self.real_mask_files:
#                     mask_name = self.real_mask_files[key]
#                     break
#
#         if img_name is None or mask_name is None:
#             # 返回空数据跳过
#             print(f"⚠️ Warning: Missing pair for {img_id}")
#             return torch.zeros(3, config['input_h'], config['input_w']), torch.zeros(config['input_h'],
#                                                                                      config['input_w']), img_id
#
#         # 3. 读取与预处理
#         img_path = os.path.join(self.img_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         image = np.array(Image.open(img_path).convert('RGB'))
#         mask = np.array(Image.open(mask_path).convert('L'))
#
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask, img_id
#
#
# def main():
#     # 设置设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"✅ 使用设备: {device}")
#
#     # ------------------------------------------------------------------
#     # 1. 实例化模型
#     # ------------------------------------------------------------------
#     print(f"=> 正在创建模型: {config['arch']}")
#
#     try:
#         # 直接使用 archs 中对应的类
#         model = archs.__dict__[config['arch']](
#             num_classes=config['num_classes'],
#             input_channels=config['input_channels'],
#             deep_supervision=config['deep_supervision'],
#             embed_dims=config['embed_dims'],
#             img_size=config['input_h']
#         ).to(device)
#     except KeyError:
#         print(f"❌ 错误: archs.py 中找不到类 '{config['arch']}'")
#         return
#     except Exception as e:
#         print(f"❌ 模型初始化失败: {e}")
#         return
#
#     # ------------------------------------------------------------------
#     # 2. 加载权重 (智能处理)
#     # ------------------------------------------------------------------
#     model_path = config['model_path']
#     if not os.path.exists(model_path):
#         print(f"❌ 错误: 权重文件不存在 -> {model_path}")
#         return
#
#     print(f"=> 正在加载权重: {model_path}")
#     checkpoint = torch.load(model_path, map_location=device)
#
#     # 处理 train.py 中 AveragedModel (EMA) 产生的 key 问题
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint.items():
#         # 过滤掉 EMA 的计数器
#         if 'n_averaged' in k:
#             continue
#         # 去掉 module. 前缀
#         name = k.replace('module.', '')
#         new_state_dict[name] = v
#
#     try:
#         model.load_state_dict(new_state_dict, strict=True)
#         print("✅ 权重加载成功 (Strict Mode)")
#     except RuntimeError as e:
#         print(f"⚠️ Strict加载失败，尝试非严格加载...\n错误信息摘要: {str(e)[:200]}...")
#         try:
#             model.load_state_dict(new_state_dict, strict=False)
#             print("⚠️ 权重以非严格模式加载成功 (部分层可能未加载)")
#         except Exception as e2:
#             print(f"❌ 权重加载彻底失败: {e2}")
#             return
#
#     model.eval()
#
#     # ------------------------------------------------------------------
#     # 3. 数据集准备 (🔥 防止数据泄露的核心修改)
#     # ------------------------------------------------------------------
#     images_dir = os.path.join(config['dataset_dir'], 'images')
#     masks_dir = os.path.join(config['dataset_dir'], 'masks')
#
#     # 先扫出所有的图
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [os.path.splitext(f)[0] for f in all_files_in_dir if
#                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
#
#     # 🟢 【核心修改 3】：改成 test_size=0.3，严格对齐训练时的 7:3 划分！
#     _, test_ids = train_test_split(valid_ids, test_size=0.3, random_state=config['dataseed'])
#
#     test_transform = Compose([
#         Resize(config['input_h'], config['input_w']),
#         Normalize(),
#         ToTensorV2()
#     ])
#
#     # 把切分好的 test_ids 传给数据集
#     test_ds = TestDataset(test_ids, images_dir, masks_dir, transform=test_transform)
#     test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
#
#     print(f"✅ 成功分离数据！严格测试集图片数量: {len(test_ds)}")
#
#     # ------------------------------------------------------------------
#     # 4. 推理循环
#     # ------------------------------------------------------------------
#     print("=> 开始推理与评估...")
#     results = []
#
#     with torch.no_grad():
#         for input, target, img_id in tqdm(test_loader):
#             input = input.to(device)
#             target = target.to(device)
#
#             # 推理
#             outputs = model(input)
#
#             # 处理深监督返回的列表 [final, aux1, aux2...]
#             if isinstance(outputs, (list, tuple)):
#                 output = outputs[0]
#             else:
#                 output = outputs
#
#             # 如果输出尺寸不匹配，进行插值
#             if output.shape[2:] != target.shape[1:]:
#                 output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             # 计算指标
#             try:
#                 # 尝试调用 metrics.py 的完整指标
#                 iou, dice, hd95, recall, specificity, precision = indicators(output, target, compute_hd95=True)
#             except:
#                 # 回退
#                 iou, _, _, _, _, _ = indicators(output, target)
#                 dice, hd95, recall, precision = 0, 0, 0, 0
#
#             # 记录结果
#             if isinstance(img_id, tuple): img_id = img_id[0]
#
#             results.append({
#                 'Image': img_id,
#                 'IoU': iou,
#                 'Dice': dice,
#                 'HD95': hd95,
#                 'Precision': precision,
#                 'Recall': recall
#             })
#
#     # ------------------------------------------------------------------
#     # 5. 结果保存
#     # ------------------------------------------------------------------
#     if len(results) > 0:
#         df = pd.DataFrame(results)
#         avg_row = df.mean(numeric_only=True)
#
#         print("\n" + "=" * 40)
#         print("       🏆 最终测试结果 (Average) 🏆")
#         print("=" * 40)
#         print(f"✅ mIoU       : {avg_row['IoU']:.4f}")
#         print(f"✅ mDice      : {avg_row['Dice']:.4f}")
#         print(f"✅ mRecall    : {avg_row['Recall']:.4f}")
#         print(f"✅ mPrecision : {avg_row['Precision']:.4f}")
#         print(f"✅ mHD95      : {avg_row['HD95']:.4f}")
#         print("=" * 40)
#
#         # 自动保存到模型同一目录下
#         save_dir = os.path.dirname(config['model_path'])
#         save_path = os.path.join(save_dir, 'final_test_results.csv')
#         df.to_csv(save_path, index=False)
#         print(f"📂 详细测试报告已保存至: {save_path}")
#     else:
#         print("❌ 未生成任何结果，请检查数据路径是否正确。")
#
#
# if __name__ == '__main__':
#     main()







#正宗test测试文件
# import os
# import sys
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import cv2
# from glob import glob
# from tqdm import tqdm
# from collections import OrderedDict
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
#
# from sklearn.model_selection import train_test_split
# from albumentations import Compose, Resize, Normalize
# from albumentations.pytorch import ToTensorV2
#
# import archs
#
# try:
#     from metrics import indicators
# except ImportError:
#     print("⚠️ 警告: 未找到 metrics.py，将使用内置的基础 IoU 计算函数")
#
#     def indicators(output, target, compute_hd95=False):
#         pred = torch.argmax(output, dim=1).cpu().numpy()
#         target = target.cpu().numpy()
#         intersection = np.logical_and(target == pred, target > 0).sum()
#         union = np.logical_or(target == pred, target > 0).sum()
#         iou = (intersection + 1e-6) / (union + 1e-6)
#         return iou, 0, 0, 0, 0, 0
#
#
# config = {
#     'arch': 'SegNet',
#     'embed_dims': [32, 64, 128],
#     'num_classes': 8,
#     'input_channels': 3,
#     'deep_supervision': False,
#     'input_h': 256,
#     'input_w': 256,
#
#     'model_path': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\outputs\SegNet\model_best.pth',
#     'dataset_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs\custom',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#     'dataseed': 3407,
# }
#
#
# class TestDataset(Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, transform=None):
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir)}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir)}
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#
#         img_name = None
#         for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#             key = (img_id + ext).lower()
#             if key in self.real_img_files:
#                 img_name = self.real_img_files[key]
#                 break
#
#         mask_name = None
#         target_mask_base = img_id + self.MASK_SUFFIX
#         for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#             key = (target_mask_base + ext).lower()
#             if key in self.real_mask_files:
#                 mask_name = self.real_mask_files[key]
#                 break
#
#         if mask_name is None:
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 key = (img_id + ext).lower()
#                 if key in self.real_mask_files:
#                     mask_name = self.real_mask_files[key]
#                     break
#
#         if img_name is None or mask_name is None:
#             print(f"⚠️ Warning: Missing pair for {img_id}")
#             return (
#                 torch.zeros(3, config['input_h'], config['input_w']),
#                 torch.zeros(config['input_h'], config['input_w']).long(),
#                 img_id
#             )
#
#         img_path = os.path.join(self.img_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         image = np.array(Image.open(img_path).convert('RGB'))
#         mask = np.array(Image.open(mask_path).convert('L'))
#
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask.long(), img_id
#
#
# def clean_state_dict_keys(checkpoint):
#     """
#     只去掉最外层 EMA/AveragedModel 带来的前缀 'module.'
#     不会误删模型内部合法层名中的 '.module.'
#     """
#     if isinstance(checkpoint, dict):
#         if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
#             checkpoint = checkpoint['state_dict']
#         elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
#             checkpoint = checkpoint['model_state_dict']
#
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint.items():
#         if 'n_averaged' in k:
#             continue
#
#         if k.startswith('module.'):
#             name = k[len('module.'):]
#         else:
#             name = k
#
#         new_state_dict[name] = v
#
#     return new_state_dict
#
#
# def smart_load_model(model, model_path, device):
#     print(f"=> 正在加载权重: {model_path}")
#     checkpoint = torch.load(model_path, map_location=device)
#     new_state_dict = clean_state_dict_keys(checkpoint)
#
#     try:
#         model.load_state_dict(new_state_dict, strict=True)
#         print("✅ 权重加载成功 (Strict Mode, 完全匹配)")
#         return True
#     except RuntimeError as e:
#         print(f"⚠️ Strict 加载失败，开始诊断...\n错误信息摘要: {str(e)[:300]}...")
#
#         incompatible = model.load_state_dict(new_state_dict, strict=False)
#
#         missing = incompatible.missing_keys
#         unexpected = incompatible.unexpected_keys
#
#         print(f"⚠️ Missing keys 数量: {len(missing)}")
#         if len(missing) > 0:
#             print("   Missing keys 示例:", missing[:10])
#
#         print(f"⚠️ Unexpected keys 数量: {len(unexpected)}")
#         if len(unexpected) > 0:
#             print("   Unexpected keys 示例:", unexpected[:10])
#
#         if len(missing) == 0 and len(unexpected) == 0:
#             print("✅ 非严格加载后检查发现权重其实已完整匹配")
#             return True
#         else:
#             print("❌ 当前模型结构与权重文件并非完全一致")
#             print("❌ 本次测试结果可能不可信，建议确认：")
#             print("   1) arch 名称是否与训练时一致")
#             print("   2) embed_dims / num_classes / input_channels 是否一致")
#             print("   3) 训练后是否又改过 archs.py 里的模型结构")
#             return False
#
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"✅ 使用设备: {device}")
#
#     print(f"=> 正在创建模型: {config['arch']}")
#
#     try:
#         model = archs.__dict__[config['arch']](
#             num_classes=config['num_classes'],
#             input_channels=config['input_channels'],
#             deep_supervision=config['deep_supervision'],
#             embed_dims=config['embed_dims'],
#             img_size=config['input_h']
#         ).to(device)
#     except KeyError:
#         print(f"❌ 错误: archs.py 中找不到类 '{config['arch']}'")
#         return
#     except Exception as e:
#         print(f"❌ 模型初始化失败: {e}")
#         return
#
#     model_path = config['model_path']
#     if not os.path.exists(model_path):
#         print(f"❌ 错误: 权重文件不存在 -> {model_path}")
#         return
#
#     load_ok = smart_load_model(model, model_path, device)
#     model.eval()
#
#     images_dir = os.path.join(config['dataset_dir'], 'images')
#     masks_dir = os.path.join(config['dataset_dir'], 'masks')
#
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [
#         os.path.splitext(f)[0] for f in all_files_in_dir
#         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
#     ]
#
#     _, test_ids = train_test_split(valid_ids, test_size=0.3, random_state=config['dataseed'])
#
#     test_transform = Compose([
#         Resize(config['input_h'], config['input_w']),
#         Normalize(),
#         ToTensorV2()
#     ])
#
#     test_ds = TestDataset(test_ids, images_dir, masks_dir, transform=test_transform)
#     test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
#
#     print(f"✅ 成功分离数据！严格测试集图片数量: {len(test_ds)}")
#
#     if not load_ok:
#         print("⚠️ 警告：当前权重没有与模型严格匹配，下面虽然继续测试，但结果不建议作为最终结论。")
#
#     print("=> 开始推理与评估...")
#     results = []
#
#     with torch.no_grad():
#         for input, target, img_id in tqdm(test_loader):
#             input = input.to(device)
#             target = target.to(device)
#
#             outputs = model(input)
#
#             if isinstance(outputs, (list, tuple)):
#                 output = outputs[0]
#             else:
#                 output = outputs
#
#             if output.shape[2:] != target.shape[1:]:
#                 output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             try:
#                 iou, dice, hd95, recall, specificity, precision = indicators(output, target, compute_hd95=True)
#             except:
#                 iou, _, _, _, _, _ = indicators(output, target)
#                 dice, hd95, recall, precision = 0, 0, 0, 0
#
#             if isinstance(img_id, tuple):
#                 img_id = img_id[0]
#             elif isinstance(img_id, list):
#                 img_id = img_id[0]
#
#             results.append({
#                 'Image': img_id,
#                 'IoU': float(iou),
#                 'Dice': float(dice),
#                 'HD95': float(hd95),
#                 'Precision': float(precision),
#                 'Recall': float(recall)
#             })
#
#     if len(results) > 0:
#         df = pd.DataFrame(results)
#         avg_row = df.mean(numeric_only=True)
#
#         print("\n" + "=" * 40)
#         print("       🏆 最终测试结果 (Average) 🏆")
#         print("=" * 40)
#         print(f"✅ mIoU       : {avg_row['IoU']:.4f}")
#         print(f"✅ mDice      : {avg_row['Dice']:.4f}")
#         print(f"✅ mRecall    : {avg_row['Recall']:.4f}")
#         print(f"✅ mPrecision : {avg_row['Precision']:.4f}")
#         print(f"✅ mHD95      : {avg_row['HD95']:.4f}")
#         print("=" * 40)
#
#         save_dir = os.path.dirname(config['model_path'])
#         save_path = os.path.join(save_dir, 'final_test_results.csv')
#         df.to_csv(save_path, index=False)
#         print(f"📂 详细测试报告已保存至: {save_path}")
#     else:
#         print("❌ 未生成任何结果，请检查数据路径是否正确。")
#
#
# if __name__ == '__main__':
#     main()






#加了BIoU的test文件基于上免得文件改的 我的论文的test文件
# import os
# import cv2
# import torch
# import archs
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F
#
# from PIL import Image
# from tqdm import tqdm
# from collections import OrderedDict
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, DataLoader
# from albumentations import Compose, Resize, Normalize
# from albumentations.pytorch import ToTensorV2
#
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
#
# try:
#     from metrics import indicators
# except ImportError:
#     print("⚠️ 警告: 未找到 metrics.py，将使用内置的基础 IoU 计算函数")
#
#     def indicators(output, target, compute_hd95=False):
#         pred = torch.argmax(output, dim=1).cpu().numpy()
#         target = target.cpu().numpy()
#         intersection = np.logical_and(target == pred, target > 0).sum()
#         union = np.logical_or(target == pred, target > 0).sum()
#         iou = (intersection + 1e-6) / (union + 1e-6)
#         return iou, 0, 0, 0, 0, 0
#
#
# config = {
#     # ===== 这里要和你的 train 保持一致 =====
#     'arch': 'MSHNet_Official',
#     'embed_dims': [32, 64, 128],   # 对应 train 里的 input_list
#     'num_classes': 8,
#     'input_channels': 3,
#     'deep_supervision': True,
#     'input_h': 256,
#     'input_w': 256,
#
#     # ===== 改成你实际训练输出的权重路径 =====
#     'model_path': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\outputs\MarFS-hfmd-add\model_best.pth',
#     'dataset_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs\custom',
#
#     'img_ext': '.png',
#     'mask_ext': '.png',
#     'dataseed': 3407,
#
#     # ===== BIoU 参数 =====
#     'boundary_dilation_ratio': 0.02,   # 一般 0.01~0.02 都可以
#     'ignore_index': 0,                 # 默认不统计背景类
# }
#
#
# class TestDataset(Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, transform=None):
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.MASK_SUFFIX = "_pure_mask_single"
#
#         self.real_img_files = {f.lower(): f for f in os.listdir(img_dir)}
#         self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir)}
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#
#         img_name = None
#         for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#             key = (img_id + ext).lower()
#             if key in self.real_img_files:
#                 img_name = self.real_img_files[key]
#                 break
#
#         mask_name = None
#         target_mask_base = img_id + self.MASK_SUFFIX
#         for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#             key = (target_mask_base + ext).lower()
#             if key in self.real_mask_files:
#                 mask_name = self.real_mask_files[key]
#                 break
#
#         if mask_name is None:
#             for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
#                 key = (img_id + ext).lower()
#                 if key in self.real_mask_files:
#                     mask_name = self.real_mask_files[key]
#                     break
#
#         if img_name is None or mask_name is None:
#             print(f"⚠️ Warning: Missing pair for {img_id}")
#             return (
#                 torch.zeros(3, config['input_h'], config['input_w']),
#                 torch.zeros(config['input_h'], config['input_w']).long(),
#                 img_id
#             )
#
#         img_path = os.path.join(self.img_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)
#
#         image = np.array(Image.open(img_path).convert('RGB'))
#         mask = np.array(Image.open(mask_path).convert('L'))
#
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#         else:
#             image = cv2.resize(image, (config['input_w'], config['input_h']))
#             mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
#             image = image.astype('float32') / 255.0
#             image = image.transpose(2, 0, 1)
#             image = torch.from_numpy(image)
#             mask = torch.from_numpy(mask).long()
#
#         return image, mask.long(), img_id
#
#
# def clean_state_dict_keys(checkpoint):
#     """
#     只去掉最外层 EMA/AveragedModel 带来的前缀 'module.'
#     不会误删模型内部合法层名中的 '.module.'
#     """
#     if isinstance(checkpoint, dict):
#         if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
#             checkpoint = checkpoint['state_dict']
#         elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
#             checkpoint = checkpoint['model_state_dict']
#
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint.items():
#         if 'n_averaged' in k:
#             continue
#
#         if k.startswith('module.'):
#             name = k[len('module.'):]
#         else:
#             name = k
#
#         new_state_dict[name] = v
#
#     return new_state_dict
#
#
# def smart_load_model(model, model_path, device):
#     print(f"=> 正在加载权重: {model_path}")
#     checkpoint = torch.load(model_path, map_location=device)
#     new_state_dict = clean_state_dict_keys(checkpoint)
#
#     try:
#         model.load_state_dict(new_state_dict, strict=True)
#         print("✅ 权重加载成功 (Strict Mode, 完全匹配)")
#         return True
#     except RuntimeError as e:
#         print(f"⚠️ Strict 加载失败，开始诊断...\n错误信息摘要: {str(e)[:300]}...")
#
#         incompatible = model.load_state_dict(new_state_dict, strict=False)
#
#         missing = incompatible.missing_keys
#         unexpected = incompatible.unexpected_keys
#
#         print(f"⚠️ Missing keys 数量: {len(missing)}")
#         if len(missing) > 0:
#             print("   Missing keys 示例:", missing[:10])
#
#         print(f"⚠️ Unexpected keys 数量: {len(unexpected)}")
#         if len(unexpected) > 0:
#             print("   Unexpected keys 示例:", unexpected[:10])
#
#         if len(missing) == 0 and len(unexpected) == 0:
#             print("✅ 非严格加载后检查发现权重其实已完整匹配")
#             return True
#         else:
#             print("❌ 当前模型结构与权重文件并非完全一致")
#             print("❌ 本次测试结果可能不可信，建议确认：")
#             print("   1) arch 名称是否与训练时一致")
#             print("   2) embed_dims / num_classes / input_channels / deep_supervision 是否一致")
#             print("   3) 训练后是否又改过 archs.py 里的模型结构")
#             return False
#
#
# def mask_to_boundary(binary_mask, dilation_ratio=0.02):
#     """
#     将二值 mask 转成 boundary mask
#     """
#     binary_mask = binary_mask.astype(np.uint8)
#     h, w = binary_mask.shape
#
#     if binary_mask.sum() == 0:
#         return np.zeros_like(binary_mask, dtype=np.uint8)
#
#     diag_len = np.sqrt(h * h + w * w)
#     dilation = max(1, int(round(dilation_ratio * diag_len)))
#
#     padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
#     kernel = np.ones((3, 3), dtype=np.uint8)
#     eroded = cv2.erode(padded, kernel, iterations=dilation)
#     eroded = eroded[1:h + 1, 1:w + 1]
#
#     boundary = binary_mask - eroded
#     boundary = (boundary > 0).astype(np.uint8)
#     return boundary
#
#
# def compute_biou_single(pred_mask, gt_mask, num_classes, ignore_index=0, dilation_ratio=0.02):
#     """
#     单张图的多类别 Boundary IoU
#     默认跳过背景类 0
#     """
#     scores = []
#
#     for cls in range(num_classes):
#         if cls == ignore_index:
#             continue
#
#         pred_cls = (pred_mask == cls).astype(np.uint8)
#         gt_cls = (gt_mask == cls).astype(np.uint8)
#
#         # pred 和 gt 都没有这个类，就跳过
#         if pred_cls.sum() == 0 and gt_cls.sum() == 0:
#             continue
#
#         pred_boundary = mask_to_boundary(pred_cls, dilation_ratio=dilation_ratio)
#         gt_boundary = mask_to_boundary(gt_cls, dilation_ratio=dilation_ratio)
#
#         inter = np.logical_and(pred_boundary, gt_boundary).sum()
#         union = np.logical_or(pred_boundary, gt_boundary).sum()
#
#         # 极端情况下 boundary union 可能为 0，做一个兜底
#         if union == 0:
#             region_inter = np.logical_and(pred_cls, gt_cls).sum()
#             region_union = np.logical_or(pred_cls, gt_cls).sum()
#             score = (region_inter + 1e-6) / (region_union + 1e-6)
#         else:
#             score = (inter + 1e-6) / (union + 1e-6)
#
#         scores.append(score)
#
#     if len(scores) == 0:
#         return 1.0
#
#     return float(np.mean(scores))
#
#
# def compute_biou(output, target, num_classes, ignore_index=0, dilation_ratio=0.02):
#     """
#     batch 版 BIoU
#     output: [B, C, H, W]
#     target: [B, H, W]
#     """
#     pred = torch.argmax(output, dim=1).detach().cpu().numpy()
#     gt = target.detach().cpu().numpy()
#
#     batch_scores = []
#     for i in range(pred.shape[0]):
#         score = compute_biou_single(
#             pred_mask=pred[i],
#             gt_mask=gt[i],
#             num_classes=num_classes,
#             ignore_index=ignore_index,
#             dilation_ratio=dilation_ratio
#         )
#         batch_scores.append(score)
#
#     return float(np.mean(batch_scores))
#
#
# def build_model_from_config(config, device):
#     print(f"=> 正在创建模型: {config['arch']}")
#
#     if config['arch'] not in archs.__dict__:
#         raise KeyError(f"archs.py 中找不到类 '{config['arch']}'")
#
#     model_class = archs.__dict__[config['arch']]
#
#     common_kwargs = dict(
#         num_classes=config['num_classes'],
#         input_channels=config['input_channels'],
#         deep_supervision=config['deep_supervision'],
#         embed_dims=config['embed_dims']
#     )
#
#     # 有些模型构造函数有 img_size，有些没有，所以这里做兼容
#     try:
#         model = model_class(
#             **common_kwargs,
#             img_size=config['input_h']
#         ).to(device)
#     except TypeError:
#         model = model_class(
#             **common_kwargs
#         ).to(device)
#
#     return model
#
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"✅ 使用设备: {device}")
#
#     try:
#         model = build_model_from_config(config, device)
#     except KeyError as e:
#         print(f"❌ 错误: {e}")
#         return
#     except Exception as e:
#         print(f"❌ 模型初始化失败: {e}")
#         return
#
#     model_path = config['model_path']
#     if not os.path.exists(model_path):
#         print(f"❌ 错误: 权重文件不存在 -> {model_path}")
#         return
#
#     load_ok = smart_load_model(model, model_path, device)
#     model.eval()
#
#     images_dir = os.path.join(config['dataset_dir'], 'images')
#     masks_dir = os.path.join(config['dataset_dir'], 'masks')
#
#     all_files_in_dir = os.listdir(images_dir)
#     valid_ids = [
#         os.path.splitext(f)[0] for f in all_files_in_dir
#         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
#     ]
#
#     # 如果你想让 test 和 train 的划分更稳定，建议 train 和 test 两边都加 sorted(...)
#     valid_ids = sorted(valid_ids)
#
#     _, test_ids = train_test_split(valid_ids, test_size=0.3, random_state=config['dataseed'])
#
#     test_transform = Compose([
#         Resize(config['input_h'], config['input_w']),
#         Normalize(),
#         ToTensorV2()
#     ])
#
#     test_ds = TestDataset(test_ids, images_dir, masks_dir, transform=test_transform)
#     test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
#
#     print(f"✅ 成功分离数据！严格测试集图片数量: {len(test_ds)}")
#
#     if not load_ok:
#         print("⚠️ 警告：当前权重没有与模型严格匹配，下面虽然继续测试，但结果不建议作为最终结论。")
#
#     print("=> 开始推理与评估...")
#     results = []
#
#     with torch.no_grad():
#         for input, target, img_id in tqdm(test_loader):
#             input = input.to(device)
#             target = target.to(device)
#
#             outputs = model(input)
#
#             if isinstance(outputs, (list, tuple)):
#                 output = outputs[0]
#             else:
#                 output = outputs
#
#             if output.shape[2:] != target.shape[1:]:
#                 output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=False)
#
#             try:
#                 iou, dice, hd95, recall, specificity, precision = indicators(output, target, compute_hd95=True)
#             except Exception:
#                 iou, _, _, _, _, _ = indicators(output, target)
#                 dice, hd95, recall, specificity, precision = 0, 0, 0, 0, 0
#
#             biou = compute_biou(
#                 output=output,
#                 target=target,
#                 num_classes=config['num_classes'],
#                 ignore_index=config['ignore_index'],
#                 dilation_ratio=config['boundary_dilation_ratio']
#             )
#
#             if isinstance(img_id, tuple):
#                 img_id = img_id[0]
#             elif isinstance(img_id, list):
#                 img_id = img_id[0]
#
#             results.append({
#                 'Image': img_id,
#                 'IoU': float(iou),
#                 'BIoU': float(biou),
#                 'Dice': float(dice),
#                 'HD95': float(hd95),
#                 'Precision': float(precision),
#                 'Recall': float(recall)
#             })
#
#     if len(results) > 0:
#         df = pd.DataFrame(results)
#         avg_row = df.mean(numeric_only=True)
#
#         print("\n" + "=" * 45)
#         print("         🏆 最终测试结果 (Average) 🏆")
#         print("=" * 45)
#         print(f"✅ mIoU       : {avg_row['IoU']:.4f}")
#         print(f"✅ mBIoU      : {avg_row['BIoU']:.4f}")
#         print(f"✅ mDice      : {avg_row['Dice']:.4f}")
#         print(f"✅ mRecall    : {avg_row['Recall']:.4f}")
#         print(f"✅ mPrecision : {avg_row['Precision']:.4f}")
#         print(f"✅ mHD95      : {avg_row['HD95']:.4f}")
#         print("=" * 45)
#
#         save_dir = os.path.dirname(config['model_path'])
#         save_path = os.path.join(save_dir, 'final_test_results_with_biou.csv')
#         df.to_csv(save_path, index=False)
#         print(f"📂 详细测试报告已保存至: {save_path}")
#     else:
#         print("❌ 未生成任何结果，请检查数据路径是否正确。")
#
#
# if __name__ == '__main__':
#     main()
#
#
#




#
#
# # test_busi_unet_binary.py
# # 作用：加载 train_busi_unet_binary.py 训练好的权重，并输出二值预测图
# # 输出结果：每张测试图对应一个 0/255 的黑白病灶分割预测图
#
# import os
# from pathlib import Path
# import warnings
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
#
# import torch
# import torch.nn.functional as F
#
# from albumentations import Compose, Resize, Normalize
# from albumentations.pytorch import ToTensorV2
#
# import archs
#
# warnings.filterwarnings("ignore")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ========================= 1. 测试配置区域，主要改这里 =========================
# test_config = {
#     # 训练好的权重位置
#     "weight_path": r"outputs\UNet_BUSI_binary\model_best.pth",
#
#     # 测试集图片文件夹
#     # 如果你没有单独测试集，就先用原 BUSI images 文件夹，并配合下面 test_id_txt 读取验证集ID
#     "test_images_dir": r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\medical_test\cvc_6\images",
#
#     # 测试集mask文件夹：有mask就填，用于计算 IoU/Dice；没有mask就填 None
#     "test_masks_dir": r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\medical_test\cvc_6\masks",
#
#     # 测试ID文件：
#     # 1）如果想测试 train.py 自动划分出来的验证集，填 outputs\UNet_BUSI_binary\val_ids.txt
#     # 2）如果想测试 test_images_dir 中全部图片，填 None
#     "test_id_txt": None,
#
#     # 输出二值预测图文件夹
#     "pred_output_dir": r"outputs\UNet_BUSI_binary\pred_binary_",
#
#     # 预测阈值，大于该值判定为病灶
#     "threshold": 0.5,
#
#     # mask命名规则：malignant (1) 对应 malignant (1)_mask
#     "mask_suffix": "",
# }
#
#
# # ========================= 2. 默认模型配置：如果权重里保存了config，会自动覆盖 =========================
# model_config = {
#     "arch": "UNet_Classic",
#     "deep_supervision": True,
#     "input_channels": 3,
#     "num_classes": 1,
#     "input_list": [32, 64, 128],
#     "input_h": 256,
#     "input_w": 256,
# }
#
#
# def build_model(cfg):
#     model_cls = archs.__dict__[cfg["arch"]]
#
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
# def clean_state_dict(state_dict):
#     """
#     兼容三种保存方式：
#     1. {"model": state_dict, ...}
#     2. 直接保存 model.state_dict()
#     3. 直接保存 ema_model.state_dict()，这时可能带 module. 和 n_averaged
#     """
#     if "model" in state_dict and isinstance(state_dict["model"], dict):
#         state_dict = state_dict["model"]
#
#     new_state = {}
#     for k, v in state_dict.items():
#         if k == "n_averaged":
#             continue
#         if k.startswith("module."):
#             k = k[len("module."):]
#         new_state[k] = v
#     return new_state
#
#
# @torch.no_grad()
# def binary_iou_dice_from_numpy(pred, mask, eps=1e-7):
#     pred = (pred > 127).astype(np.float32)
#     mask = (mask > 127).astype(np.float32)
#
#     intersection = (pred * mask).sum()
#     union = pred.sum() + mask.sum() - intersection
#
#     iou = (intersection + eps) / (union + eps)
#     dice = (2 * intersection + eps) / (pred.sum() + mask.sum() + eps)
#     return float(iou), float(dice)
#
#
# def find_file_by_stem(folder, stem):
#     folder = Path(folder)
#     exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
#     file_map = {
#         p.name.lower(): p.name
#         for p in folder.iterdir()
#         if p.is_file() and p.suffix.lower() in exts
#     }
#
#     for ext in exts:
#         key = (stem + ext).lower()
#         if key in file_map:
#             return folder / file_map[key]
#     return None
#
#
# def load_test_ids(test_images_dir, test_id_txt=None):
#     test_images_dir = Path(test_images_dir)
#     exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
#
#     if test_id_txt is not None and str(test_id_txt).strip() != "":
#         test_id_txt = Path(test_id_txt)
#         if test_id_txt.exists():
#             with open(test_id_txt, "r", encoding="utf-8") as f:
#                 ids = [line.strip() for line in f.readlines() if line.strip()]
#             print(f"📌 从ID文件读取测试图像: {len(ids)} 张")
#             return ids
#         else:
#             print(f"⚠️ 未找到 test_id_txt: {test_id_txt}，将测试文件夹内全部图片")
#
#     ids = [p.stem for p in test_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
#     print(f"📌 从测试文件夹读取全部图像: {len(ids)} 张")
#     return ids
#
#
# def main():
#     print(f"✅ Device: {device}")
#
#     weight_path = Path(test_config["weight_path"])
#     if not weight_path.exists():
#         raise FileNotFoundError(f"找不到权重文件: {weight_path}")
#
#     checkpoint = torch.load(weight_path, map_location=device)
#
#     # 优先读取训练时保存的config，保证test模型结构与train完全一致
#     cfg = model_config.copy()
#     if isinstance(checkpoint, dict) and "config" in checkpoint:
#         train_cfg = checkpoint["config"]
#         for k in cfg.keys():
#             if k in train_cfg:
#                 cfg[k] = train_cfg[k]
#
#     print("📌 使用模型配置:")
#     for k, v in cfg.items():
#         print(f"   {k}: {v}")
#
#     model = build_model(cfg)
#     state = clean_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint)
#     missing, unexpected = model.load_state_dict(state, strict=False)
#
#     if len(missing) > 0:
#         print(f"⚠️ missing keys 数量: {len(missing)}")
#     if len(unexpected) > 0:
#         print(f"⚠️ unexpected keys 数量: {len(unexpected)}")
#
#     model.eval()
#
#     test_images_dir = Path(test_config["test_images_dir"])
#     pred_output_dir = Path(test_config["pred_output_dir"])
#     pred_output_dir.mkdir(parents=True, exist_ok=True)
#
#     test_ids = load_test_ids(test_images_dir, test_config.get("test_id_txt", None))
#
#     tf = Compose([
#         Resize(cfg["input_h"], cfg["input_w"]),
#         Normalize(),
#         ToTensorV2(),
#     ])
#
#     metrics_list = []
#
#     for img_id in tqdm(test_ids, desc="Testing"):
#         img_path = find_file_by_stem(test_images_dir, img_id)
#         if img_path is None:
#             print(f"⚠️ 找不到测试图像: {img_id}")
#             continue
#
#         image_pil = Image.open(img_path).convert("RGB")
#         orig_w, orig_h = image_pil.size
#         image_np = np.array(image_pil)
#
#         augmented = tf(image=image_np)
#         image_tensor = augmented["image"].float().unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             outputs = model(image_tensor)
#
#         logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
#         prob = torch.sigmoid(logits)
#
#         # 恢复到原图尺寸
#         prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
#         prob_np = prob.squeeze().detach().cpu().numpy()
#
#         pred_bin = (prob_np > test_config["threshold"]).astype(np.uint8) * 255
#
#         # 保存二值预测图
#         out_name = f"{img_id}_pred.png"
#         out_path = pred_output_dir / out_name
#         Image.fromarray(pred_bin).save(out_path)
#
#         # 如果有mask，则计算 IoU/Dice
#         if test_config.get("test_masks_dir", None):
#             mask_path = find_file_by_stem(
#                 test_config["test_masks_dir"],
#                 img_id + test_config["mask_suffix"]
#             )
#             if mask_path is None:
#                 mask_path = find_file_by_stem(test_config["test_masks_dir"], img_id)
#
#             if mask_path is not None:
#                 mask = np.array(Image.open(mask_path).convert("L"))
#                 if mask.shape != pred_bin.shape:
#                     mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
#
#                 iou, dice = binary_iou_dice_from_numpy(pred_bin, mask)
#                 metrics_list.append({
#                     "image_id": img_id,
#                     "iou": iou,
#                     "dice": dice,
#                     "pred_path": str(out_path),
#                 })
#
#     if len(metrics_list) > 0:
#         df = pd.DataFrame(metrics_list)
#         csv_path = pred_output_dir / "test_metrics.csv"
#         df.to_csv(csv_path, index=False, encoding="utf-8-sig")
#
#         print("\n✅ 测试完成")
#         print(f"📁 二值预测图输出文件夹: {pred_output_dir}")
#         print(f"📁 测试指标CSV: {csv_path}")
#         print(f"📌 Mean IoU : {df['iou'].mean():.4f}")
#         print(f"📌 Mean Dice: {df['dice'].mean():.4f}")
#     else:
#         print("\n✅ 测试完成")
#         print(f"📁 二值预测图输出文件夹: {pred_output_dir}")
#         print("⚠️ 没有计算指标，因为未提供mask或未匹配到mask。")
#
#
# if __name__ == "__main__":
#     main()







# test_busi_cvc_unet_binary_safe.py
# 作用：加载训练好的二值分割权重，对自定义测试集 images 文件夹中的图片进行预测
# 重点修复：
# 1. 不再读取 val_ids.txt，而是自动读取测试文件夹全部图片
# 2. 支持 images 和 masks 同名，例如 busi_1.png 对应 busi_1.png，cvc_1.png 对应 cvc_1.png
# 3. 使用更稳健的 cv2.imdecode 读取图片，遇到损坏图片会跳过并提示，不会直接崩溃
# 4. 会输出二值预测图和 test_metrics.csv

import os
from pathlib import Path
import warnings

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

import archs

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================= 1. 测试配置区域，主要改这里 =========================
test_config = {
    # 训练好的权重位置
    "weight_path": r"D:\Users\pc\PycharmProjects\PythonProject2\HHX-USV\outputs\UNet_cvc_binary\model_best.pth",

    # 如果测试 BUSI，就用 busi_6
    # "test_images_dir": r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\medical_test\busi_6\images",
    # "test_masks_dir":  r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\medical_test\busi_6\masks",

    # 如果测试 CVC，就用 cvc_6
    "test_images_dir": r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\medical_test\cvc_6\images",
    "test_masks_dir":  r"D:\Users\pc\PycharmProjects\PythonProject2\U-KAN-main\Seg_UKAN\medical_test\cvc_6\masks",

    # 关键：自定义测试集必须设置为 None，让程序自动读 images 里的 cvc_1、cvc_2...
    "test_id_txt": None,

    # 输出二值预测图文件夹
    "pred_output_dir": r"outputs\UNet_cvc_binary\pred_binary_custom",

    # 预测阈值
    "threshold": 0.5,

    # 你的测试集标签与图片同名，所以设置为空字符串
    # 例如 images\cvc_1.png 对应 masks\cvc_1.png
    "mask_suffix": "",
}


# ========================= 2. 默认模型配置：如果权重里保存了 config，会自动覆盖 =========================
model_config = {
    "arch": "UNet_Classic",
    "deep_supervision": True,
    "input_channels": 3,
    "num_classes": 1,
    "input_list": [32, 64, 128],
    "input_h": 256,
    "input_w": 256,
}


def safe_read_rgb_image(img_path):
    """
    稳健读取图像。
    返回：
    image_rgb: RGB格式numpy数组，读取失败则返回 None
    """
    img_path = Path(img_path)

    if not img_path.exists():
        print(f"⚠️ 文件不存在: {img_path}")
        return None

    if img_path.stat().st_size == 0:
        print(f"⚠️ 空文件，已跳过: {img_path}")
        return None

    try:
        # 用 imdecode 比 cv2.imread 更稳，兼容部分特殊路径
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print(f"⚠️ cv2无法识别该图片，可能文件损坏或不是真正图片: {img_path}")
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    except Exception as e:
        print(f"⚠️ 读取图片失败: {img_path}")
        print(f"   错误信息: {e}")
        return None


def safe_read_gray_image(mask_path):
    """
    稳健读取灰度mask。
    返回：
    mask: 灰度numpy数组，读取失败则返回 None
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        return None

    if mask_path.stat().st_size == 0:
        print(f"⚠️ 空mask文件，已跳过: {mask_path}")
        return None

    try:
        data = np.fromfile(str(mask_path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"⚠️ cv2无法识别该mask，可能文件损坏或不是真正图片: {mask_path}")
            return None

        return mask

    except Exception as e:
        print(f"⚠️ 读取mask失败: {mask_path}")
        print(f"   错误信息: {e}")
        return None


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


def clean_state_dict(state_dict):
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    new_state = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    return new_state


def binary_iou_dice_from_numpy(pred, mask, eps=1e-7):
    pred = (pred > 127).astype(np.float32)
    mask = (mask > 127).astype(np.float32)

    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred.sum() + mask.sum() + eps)

    return float(iou), float(dice)


def get_valid_file_list(folder):
    folder = Path(folder)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    if not folder.exists():
        raise FileNotFoundError(f"找不到文件夹: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    return files


def find_file_by_stem(folder, stem):
    files = get_valid_file_list(folder)
    file_map = {p.stem.lower(): p for p in files}
    return file_map.get(stem.lower(), None)


def find_mask_file(mask_dir, img_id, mask_suffix=""):
    if mask_dir is None or str(mask_dir).strip() == "":
        return None

    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        print(f"⚠️ mask文件夹不存在: {mask_dir}")
        return None

    candidates = []

    if mask_suffix is not None:
        candidates.append(img_id + mask_suffix)

    candidates.extend([
        img_id,
        img_id + "_mask",
        img_id + "_mask_1",
    ])

    unique_candidates = []
    for c in candidates:
        if c not in unique_candidates:
            unique_candidates.append(c)

    for c in unique_candidates:
        p = find_file_by_stem(mask_dir, c)
        if p is not None:
            return p

    return None


def load_test_ids(test_images_dir, test_id_txt=None):
    test_images_dir = Path(test_images_dir)

    if test_id_txt is not None and str(test_id_txt).strip() != "":
        test_id_txt = Path(test_id_txt)
        if test_id_txt.exists():
            with open(test_id_txt, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f.readlines() if line.strip()]
            print(f"📌 从ID文件读取测试图像: {len(ids)} 张")
            return ids
        else:
            print(f"⚠️ 未找到 test_id_txt: {test_id_txt}，将测试文件夹内全部图片")

    files = get_valid_file_list(test_images_dir)
    ids = [p.stem for p in files]

    def sort_key(x):
        try:
            return int(x.split("_")[-1])
        except:
            return x

    ids = sorted(ids, key=sort_key)

    print(f"📌 从测试文件夹读取全部图像: {len(ids)} 张")
    print(f"📌 测试图像ID: {ids}")
    return ids


def main():
    print(f"✅ Device: {device}")

    weight_path = Path(test_config["weight_path"])
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device)

    cfg = model_config.copy()
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        train_cfg = checkpoint["config"]
        for k in cfg.keys():
            if k in train_cfg:
                cfg[k] = train_cfg[k]

    print("📌 使用模型配置:")
    for k, v in cfg.items():
        print(f"   {k}: {v}")

    model = build_model(cfg)
    state = clean_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if len(missing) > 0:
        print(f"⚠️ missing keys 数量: {len(missing)}")
    if len(unexpected) > 0:
        print(f"⚠️ unexpected keys 数量: {len(unexpected)}")

    model.eval()

    test_images_dir = Path(test_config["test_images_dir"])
    pred_output_dir = Path(test_config["pred_output_dir"])
    pred_output_dir.mkdir(parents=True, exist_ok=True)

    test_ids = load_test_ids(test_images_dir, test_config.get("test_id_txt", None))

    tf = Compose([
        Resize(cfg["input_h"], cfg["input_w"]),
        Normalize(),
        ToTensorV2(),
    ])

    metrics_list = []
    failed_list = []

    for img_id in tqdm(test_ids, desc="Testing"):
        img_path = find_file_by_stem(test_images_dir, img_id)
        if img_path is None:
            print(f"⚠️ 找不到测试图像: {img_id}")
            failed_list.append({"image_id": img_id, "reason": "找不到测试图像"})
            continue

        image_np = safe_read_rgb_image(img_path)
        if image_np is None:
            failed_list.append({"image_id": img_id, "image_path": str(img_path), "reason": "图像文件无法识别"})
            continue

        orig_h, orig_w = image_np.shape[:2]

        augmented = tf(image=image_np)
        image_tensor = augmented["image"].float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        prob = torch.sigmoid(logits)

        prob = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        prob_np = prob.squeeze().detach().cpu().numpy()

        pred_bin = (prob_np > test_config["threshold"]).astype(np.uint8) * 255

        out_name = f"{img_id}_pred.png"
        out_path = pred_output_dir / out_name
        Image.fromarray(pred_bin).save(out_path)

        mask_path = find_mask_file(
            test_config.get("test_masks_dir", None),
            img_id,
            test_config.get("mask_suffix", "")
        )

        if mask_path is not None:
            mask = safe_read_gray_image(mask_path)

            if mask is not None:
                if mask.shape != pred_bin.shape:
                    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                iou, dice = binary_iou_dice_from_numpy(pred_bin, mask)

                metrics_list.append({
                    "image_id": img_id,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "iou": iou,
                    "dice": dice,
                    "pred_path": str(out_path),
                })
            else:
                print(f"⚠️ {img_id} 的mask无法识别，只保存预测图，不计算IoU/Dice")
        else:
            print(f"⚠️ 未找到 {img_id} 的mask，只保存预测图，不计算IoU/Dice")

    if len(metrics_list) > 0:
        df = pd.DataFrame(metrics_list)
        csv_path = pred_output_dir / "test_metrics.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print("\n✅ 测试完成")
        print(f"📁 二值预测图输出文件夹: {pred_output_dir}")
        print(f"📁 测试指标CSV: {csv_path}")
        print(f"📌 Mean IoU : {df['iou'].mean():.4f}")
        print(f"📌 Mean Dice: {df['dice'].mean():.4f}")
    else:
        print("\n✅ 测试完成")
        print(f"📁 二值预测图输出文件夹: {pred_output_dir}")
        print("⚠️ 没有计算指标，因为未提供mask、未匹配到mask，或图片无法识别。")

    if len(failed_list) > 0:
        failed_csv = pred_output_dir / "failed_images.csv"
        pd.DataFrame(failed_list).to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(f"⚠️ 有 {len(failed_list)} 张图片读取失败，详情见: {failed_csv}")


if __name__ == "__main__":
    main()
