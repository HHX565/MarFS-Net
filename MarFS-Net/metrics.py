import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def iou_score(output, target, compute_hd95=False):
    """
    compute_hd95: bool, 是否计算 HD95 (训练时建议 False 以提速)
    """
    if torch.is_tensor(output):
        output = torch.argmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    batch_size = output.shape[0]
    num_classes = 8

    ious = []
    precisions = []
    recalls = []
    specificities = []
    hd95s = []

    for cls in range(num_classes):
        pred_inds_all = output == cls
        target_inds_all = target == cls

        # --- 基础指标 (快速) ---
        intersection = (pred_inds_all & target_inds_all).sum()
        pred_sum = pred_inds_all.sum()
        target_sum = target_inds_all.sum()
        union = pred_sum + target_sum - intersection

        # TN 近似
        total_pixels = output.size
        tn = total_pixels - union
        fp = pred_sum - intersection

        # IoU
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))

        # Precision
        if pred_sum == 0:
            precisions.append(0.0)
        else:
            precisions.append(float(intersection) / float(pred_sum))

        # Recall
        if target_sum == 0:
            recalls.append(float('nan'))
        else:
            recalls.append(float(intersection) / float(target_sum))

        # Specificity
        if (tn + fp) == 0:
            specificities.append(float('nan'))
        else:
            specificities.append(float(tn) / float(tn + fp))

        # --- HD95 计算 (慢速，由开关控制) ---
        if compute_hd95:
            batch_hd95 = []
            for b in range(batch_size):
                pred_mask = pred_inds_all[b]
                target_mask = target_inds_all[b]

                # 只有当预测和标签都有内容时才能算距离
                if np.any(pred_mask) and np.any(target_mask):
                    pred_points = np.argwhere(pred_mask)
                    target_points = np.argwhere(target_mask)
                    d_p_t = directed_hausdorff(pred_points, target_points)[0]
                    d_t_p = directed_hausdorff(target_points, pred_points)[0]
                    batch_hd95.append(max(d_p_t, d_t_p))
                elif not np.any(pred_mask) and not np.any(target_mask):
                    batch_hd95.append(0)  # 都是背景，距离为0
                else:
                    batch_hd95.append(float('nan'))  # 一个有一个没有，无法计算

            if len(batch_hd95) > 0:
                hd95s.append(np.nanmean(batch_hd95))
            else:
                hd95s.append(float('nan'))
        else:
            hd95s.append(0)  # 不计算时设为0

    # 平均值
    miou = np.nanmean(ious)
    mprec = np.nanmean(precisions)
    mrecall = np.nanmean(recalls)
    mspec = np.nanmean(specificities)
    mhd95 = np.nanmean(hd95s) if compute_hd95 else 0

    # Dice
    dice = np.nanmean([(2 * x) / (x + 1) for x in ious if not np.isnan(x)])

    # 填充 NaN
    miou = 0 if np.isnan(miou) else miou
    mprec = 0 if np.isnan(mprec) else mprec
    mrecall = 0 if np.isnan(mrecall) else mrecall
    mspec = 0 if np.isnan(mspec) else mspec
    mhd95 = 0 if np.isnan(mhd95) else mhd95
    dice = 0 if np.isnan(dice) else dice

    return miou, dice, mhd95, mrecall, mspec, mprec


def indicators(output, target, compute_hd95=False):
    # 透传 compute_hd95 参数
    return iou_score(output, target, compute_hd95=compute_hd95)