import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
import nibabel as nib
import numpy as np
from scipy.spatial._hausdorff import directed_hausdorff


def dice_coefficient(prediction, ground_truth):
    # 计算dice系数
    intersection = np.logical_and(prediction, ground_truth)
    dice = (2. * np.sum(intersection)) / (np.sum(prediction) + np.sum(ground_truth))
    return dice


def hausdorff_distance(setA, setB):
    # 计算A中每个点到B的最小距离
    dist_A_to_B = np.sqrt(np.min(np.sum((setA[:, np.newaxis] - setB) ** 2, axis=2), axis=1))

    # 计算B中每个点到A的最小距离
    dist_B_to_A = np.sqrt(np.min(np.sum((setB[:, np.newaxis] - setA) ** 2, axis=2), axis=1))

    # Hausdorff距离为两者中较大的最小距离
    hausdorff_dist = max(np.max(dist_A_to_B), np.max(dist_B_to_A))

    return hausdorff_dist


def precision(gt_data,pred_data):
    tp = np.sum(np.logical_and(gt_data, pred_data))
    fp = np.sum(np.logical_and(np.logical_not(gt_data), pred_data))

    precision = tp / (tp + fp)

    return precision


def calculate_3d_iou(volume1, volume2):
    # 计算两个区域的重叠体积
    intersection_volume = np.sum(np.minimum(volume1, volume2))

    # 计算两个区域的总体积
    union_volume = np.sum(volume1) + np.sum(volume2) - intersection_volume

    # 计算IOU
    iou = intersection_volume / union_volume
    return iou


def recall_score(ground_truth, predicted):
    true_positive = np.sum(np.logical_and(ground_truth == 1, predicted == 1))
    false_negative = np.sum(np.logical_and(ground_truth == 1, predicted == 0))

    recall = true_positive / (true_positive + false_negative)

    return recall


def hd95_score(ground_truth, predicted):
    hd_distances = []
    for i in range(ground_truth.shape[0]):
        # 将输入数组转换为C连续的数组
        ground_truth_i = np.ascontiguousarray(ground_truth[i])
        predicted_i = np.ascontiguousarray(predicted[i])
        hd_distances.append(max(directed_hausdorff(ground_truth_i, predicted_i)[0],
                                directed_hausdorff(predicted_i, ground_truth_i)[0]))
    hd95 = np.percentile(hd_distances, 95)

    return hd95


def calculate_dice(folder1, folder2):
    dice_scores = []
    for file1 in os.listdir(folder1):
        if file1.endswith(".nii.gz"):
            file2 = os.path.join(folder2, file1)
            if os.path.isfile(file2):
                image1 = nib.load(os.path.join(folder1, file1))
                image2 = nib.load(file2)
                data1 = image1.get_fdata()
                data2 = image2.get_fdata()
                dice = dice_coefficient(data1, data2)
                dice_scores.append(dice)
                # print(f"{file1}: {dice}")
                print(f"{dice}")
    return dice_scores


def calculate_precision(folder1, folder2):
    precision_scores = []
    for file1 in os.listdir(folder1):
        if file1.endswith(".nii.gz"):
            file2 = os.path.join(folder2, file1)
            if os.path.isfile(file2):
                image1 = nib.load(os.path.join(folder1, file1))
                image2 = nib.load(file2)
                data1 = image1.get_fdata()
                data2 = image2.get_fdata()
                pre = precision(data1, data2)
                precision_scores.append(pre)
                # print(f"Dice coefficient for {file1}: {dice}")
    return precision_scores


def calculate_recall(folder1, folder2):
    recall_scores = []
    for file1 in os.listdir(folder1):
        if file1.endswith(".nii.gz"):
            file2 = os.path.join(folder2, file1)
            if os.path.isfile(file2):
                image1 = nib.load(os.path.join(folder1, file1))
                image2 = nib.load(file2)
                data1 = image1.get_fdata()
                data2 = image2.get_fdata()
                recall = recall_score(data1, data2)
                recall_scores.append(recall)
                # print(f"Dice coefficient for {file1}: {dice}")
    return recall_scores


def calculate_iou(folder1, folder2):
    iou_scores = []
    for file1 in os.listdir(folder1):
        if file1.endswith(".nii.gz"):
            file2 = os.path.join(folder2, file1)
            if os.path.isfile(file2):
                image1 = nib.load(os.path.join(folder1, file1))
                image2 = nib.load(file2)
                data1 = image1.get_fdata()
                data2 = image2.get_fdata()
                iou = calculate_3d_iou(data1, data2)
                iou_scores.append(iou)
                # print(f"Dice coefficient for {file1}: {dice}")
    return iou_scores


def calculate_hd95(folder1, folder2):
    hd95_scores = []
    for file1 in os.listdir(folder1):
        if file1.endswith(".nii.gz"):
            file2 = os.path.join(folder2, file1)
            if os.path.isfile(file2):
                image1 = nib.load(os.path.join(folder1, file1))
                image2 = nib.load(file2)
                data1 = image1.get_fdata()
                data2 = image2.get_fdata()
                hd95 = hd95_score(data1, data2)
                hd95_scores.append(hd95)
                # print(f"Dice coefficient for {file1}: {dice}")
    return hd95_scores


folder1 = r"D:\HECKTOR\model_result\TASE-UNet6\test"
folder2 = r"D:\HECKTOR\DATA_250\test_label_GTVp"

dice_scores = calculate_dice(folder1, folder2)
average_dice = np.mean(dice_scores)
print("Average Dice coefficient:", average_dice)

precision_scores = calculate_precision(folder1, folder2)
average_precision = np.mean(precision_scores)
print("Average precison coefficient:", average_precision)

# recall_scores = calculate_recall(folder1, folder2)
# average_recall = np.mean(recall_scores)
# print("Average recall coefficient:", average_recall)

iou_scores = calculate_iou(folder1, folder2)
average_iou = np.mean(iou_scores)
print("Average iou coefficient:", average_iou)

hd95_scores = calculate_hd95(folder1, folder2)
average_hd95 = np.mean(hd95_scores)
print("Average hd95 coefficient:", average_hd95)