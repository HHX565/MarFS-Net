import os
import cv2
import torch
import archs
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    from metrics import indicators
except ImportError:
    print("⚠️ 警告: 未找到 metrics.py，将使用内置的基础 IoU 计算函数")

    def indicators(output, target, compute_hd95=False):
        pred = torch.argmax(output, dim=1).cpu().numpy()
        target = target.cpu().numpy()
        intersection = np.logical_and(target == pred, target > 0).sum()
        union = np.logical_or(target == pred, target > 0).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou, 0, 0, 0, 0, 0


config = {
    # ===== 这里要和你的 train 保持一致 =====
    'arch': 'MSHNet_Official',
    'embed_dims': [32, 64, 128],   # 对应 train 里的 input_list
    'num_classes': 8,
    'input_channels': 3,
    'deep_supervision': True,
    'input_h': 256,
    'input_w': 256,

    # ===== 改成你实际训练输出的权重路径 =====
    'model_path': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\outputs\MarFS-hfmd-add\model_best.pth',
    'dataset_dir': r'C:\Users\zyc\PycharmProjects\PythonProject1\HHX-USV\inputs\custom',

    'img_ext': '.png',
    'mask_ext': '.png',
    'dataseed': 3407,

    # ===== BIoU 参数 =====
    'boundary_dilation_ratio': 0.02,   # 一般 0.01~0.02 都可以
    'ignore_index': 0,                 # 默认不统计背景类
}


class TestDataset(Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.MASK_SUFFIX = "_pure_mask_single"

        self.real_img_files = {f.lower(): f for f in os.listdir(img_dir)}
        self.real_mask_files = {f.lower(): f for f in os.listdir(mask_dir)}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_name = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            key = (img_id + ext).lower()
            if key in self.real_img_files:
                img_name = self.real_img_files[key]
                break

        mask_name = None
        target_mask_base = img_id + self.MASK_SUFFIX
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            key = (target_mask_base + ext).lower()
            if key in self.real_mask_files:
                mask_name = self.real_mask_files[key]
                break

        if mask_name is None:
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                key = (img_id + ext).lower()
                if key in self.real_mask_files:
                    mask_name = self.real_mask_files[key]
                    break

        if img_name is None or mask_name is None:
            print(f"⚠️ Warning: Missing pair for {img_id}")
            return (
                torch.zeros(3, config['input_h'], config['input_w']),
                torch.zeros(config['input_h'], config['input_w']).long(),
                img_id
            )

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (config['input_w'], config['input_h']))
            mask = cv2.resize(mask, (config['input_w'], config['input_h']), interpolation=cv2.INTER_NEAREST)
            image = image.astype('float32') / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask).long()

        return image, mask.long(), img_id


def clean_state_dict_keys(checkpoint):
    """
    只去掉最外层 EMA/AveragedModel 带来的前缀 'module.'
    不会误删模型内部合法层名中的 '.module.'
    """
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            checkpoint = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            checkpoint = checkpoint['model_state_dict']

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if 'n_averaged' in k:
            continue

        if k.startswith('module.'):
            name = k[len('module.'):]
        else:
            name = k

        new_state_dict[name] = v

    return new_state_dict


def smart_load_model(model, model_path, device):
    print(f"=> 正在加载权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = clean_state_dict_keys(checkpoint)

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 权重加载成功 (Strict Mode, 完全匹配)")
        return True
    except RuntimeError as e:
        print(f"⚠️ Strict 加载失败，开始诊断...\n错误信息摘要: {str(e)[:300]}...")

        incompatible = model.load_state_dict(new_state_dict, strict=False)

        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys

        print(f"⚠️ Missing keys 数量: {len(missing)}")
        if len(missing) > 0:
            print("   Missing keys 示例:", missing[:10])

        print(f"⚠️ Unexpected keys 数量: {len(unexpected)}")
        if len(unexpected) > 0:
            print("   Unexpected keys 示例:", unexpected[:10])

        if len(missing) == 0 and len(unexpected) == 0:
            print("✅ 非严格加载后检查发现权重其实已完整匹配")
            return True
        else:
            print("❌ 当前模型结构与权重文件并非完全一致")
            print("❌ 本次测试结果可能不可信，建议确认：")
            print("   1) arch 名称是否与训练时一致")
            print("   2) embed_dims / num_classes / input_channels / deep_supervision 是否一致")
            print("   3) 训练后是否又改过 archs.py 里的模型结构")
            return False


def mask_to_boundary(binary_mask, dilation_ratio=0.02):
    """
    将二值 mask 转成 boundary mask
    """
    binary_mask = binary_mask.astype(np.uint8)
    h, w = binary_mask.shape

    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    diag_len = np.sqrt(h * h + w * w)
    dilation = max(1, int(round(dilation_ratio * diag_len)))

    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)
    eroded = eroded[1:h + 1, 1:w + 1]

    boundary = binary_mask - eroded
    boundary = (boundary > 0).astype(np.uint8)
    return boundary


def compute_biou_single(pred_mask, gt_mask, num_classes, ignore_index=0, dilation_ratio=0.02):
    """
    单张图的多类别 Boundary IoU
    默认跳过背景类 0
    """
    scores = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = (pred_mask == cls).astype(np.uint8)
        gt_cls = (gt_mask == cls).astype(np.uint8)

        # pred 和 gt 都没有这个类，就跳过
        if pred_cls.sum() == 0 and gt_cls.sum() == 0:
            continue

        pred_boundary = mask_to_boundary(pred_cls, dilation_ratio=dilation_ratio)
        gt_boundary = mask_to_boundary(gt_cls, dilation_ratio=dilation_ratio)

        inter = np.logical_and(pred_boundary, gt_boundary).sum()
        union = np.logical_or(pred_boundary, gt_boundary).sum()

        # 极端情况下 boundary union 可能为 0，做一个兜底
        if union == 0:
            region_inter = np.logical_and(pred_cls, gt_cls).sum()
            region_union = np.logical_or(pred_cls, gt_cls).sum()
            score = (region_inter + 1e-6) / (region_union + 1e-6)
        else:
            score = (inter + 1e-6) / (union + 1e-6)

        scores.append(score)

    if len(scores) == 0:
        return 1.0

    return float(np.mean(scores))


def compute_biou(output, target, num_classes, ignore_index=0, dilation_ratio=0.02):
    """
    batch 版 BIoU
    output: [B, C, H, W]
    target: [B, H, W]
    """
    pred = torch.argmax(output, dim=1).detach().cpu().numpy()
    gt = target.detach().cpu().numpy()

    batch_scores = []
    for i in range(pred.shape[0]):
        score = compute_biou_single(
            pred_mask=pred[i],
            gt_mask=gt[i],
            num_classes=num_classes,
            ignore_index=ignore_index,
            dilation_ratio=dilation_ratio
        )
        batch_scores.append(score)

    return float(np.mean(batch_scores))


def build_model_from_config(config, device):
    print(f"=> 正在创建模型: {config['arch']}")

    if config['arch'] not in archs.__dict__:
        raise KeyError(f"archs.py 中找不到类 '{config['arch']}'")

    model_class = archs.__dict__[config['arch']]

    common_kwargs = dict(
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        embed_dims=config['embed_dims']
    )

    # 有些模型构造函数有 img_size，有些没有，所以这里做兼容
    try:
        model = model_class(
            **common_kwargs,
            img_size=config['input_h']
        ).to(device)
    except TypeError:
        model = model_class(
            **common_kwargs
        ).to(device)

    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    try:
        model = build_model_from_config(config, device)
    except KeyError as e:
        print(f"❌ 错误: {e}")
        return
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    model_path = config['model_path']
    if not os.path.exists(model_path):
        print(f"❌ 错误: 权重文件不存在 -> {model_path}")
        return

    load_ok = smart_load_model(model, model_path, device)
    model.eval()

    images_dir = os.path.join(config['dataset_dir'], 'images')
    masks_dir = os.path.join(config['dataset_dir'], 'masks')

    all_files_in_dir = os.listdir(images_dir)
    valid_ids = [
        os.path.splitext(f)[0] for f in all_files_in_dir
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    # 如果你想让 test 和 train 的划分更稳定，建议 train 和 test 两边都加 sorted(...)
    valid_ids = sorted(valid_ids)

    _, test_ids = train_test_split(valid_ids, test_size=0.3, random_state=config['dataseed'])

    test_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
        ToTensorV2()
    ])

    test_ds = TestDataset(test_ids, images_dir, masks_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"✅ 成功分离数据！严格测试集图片数量: {len(test_ds)}")

    if not load_ok:
        print("⚠️ 警告：当前权重没有与模型严格匹配，下面虽然继续测试，但结果不建议作为最终结论。")

    print("=> 开始推理与评估...")
    results = []

    with torch.no_grad():
        for input, target, img_id in tqdm(test_loader):
            input = input.to(device)
            target = target.to(device)

            outputs = model(input)

            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs

            if output.shape[2:] != target.shape[1:]:
                output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=False)

            try:
                iou, dice, hd95, recall, specificity, precision = indicators(output, target, compute_hd95=True)
            except Exception:
                iou, _, _, _, _, _ = indicators(output, target)
                dice, hd95, recall, specificity, precision = 0, 0, 0, 0, 0

            biou = compute_biou(
                output=output,
                target=target,
                num_classes=config['num_classes'],
                ignore_index=config['ignore_index'],
                dilation_ratio=config['boundary_dilation_ratio']
            )

            if isinstance(img_id, tuple):
                img_id = img_id[0]
            elif isinstance(img_id, list):
                img_id = img_id[0]

            results.append({
                'Image': img_id,
                'IoU': float(iou),
                'BIoU': float(biou),
                'Dice': float(dice),
                'HD95': float(hd95),
                'Precision': float(precision),
                'Recall': float(recall)
            })

    if len(results) > 0:
        df = pd.DataFrame(results)
        avg_row = df.mean(numeric_only=True)

        print("\n" + "=" * 45)
        print("         🏆 最终测试结果 (Average) 🏆")
        print("=" * 45)
        print(f"✅ mIoU       : {avg_row['IoU']:.4f}")
        print(f"✅ mBIoU      : {avg_row['BIoU']:.4f}")
        print(f"✅ mDice      : {avg_row['Dice']:.4f}")
        print(f"✅ mRecall    : {avg_row['Recall']:.4f}")
        print(f"✅ mPrecision : {avg_row['Precision']:.4f}")
        print(f"✅ mHD95      : {avg_row['HD95']:.4f}")
        print("=" * 45)

        save_dir = os.path.dirname(config['model_path'])
        save_path = os.path.join(save_dir, 'final_test_results_with_biou.csv')
        df.to_csv(save_path, index=False)
        print(f"📂 详细测试报告已保存至: {save_path}")
    else:
        print("❌ 未生成任何结果，请检查数据路径是否正确。")


if __name__ == '__main__':
    main()

