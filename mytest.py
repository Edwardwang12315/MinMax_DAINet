#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from models.factory import build_net
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
from tqdm import tqdm

# 启用混合精度
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class FaceDetectionDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.basename(img_path)

def to_chw_bgr(image):
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    image = image[[2, 1, 0], :, :]
    return image

def preprocess_image(image, shrink=1.0):
    """预处理图像，支持批量处理"""
    if isinstance(image, np.ndarray) and len(image.shape) == 3:
        image = [image]
    
    processed_images = []
    for img in image:
        # 调整大小
        if shrink != 1.0:
            img = cv2.resize(img, None, None, fx=shrink, fy=shrink, 
                            interpolation=cv2.INTER_LINEAR)
        
        # 转换格式
        x = to_chw_bgr(img)
        x = x.astype('float32') / 255.
        x = x[[2, 1, 0], :, :]  # BGR to RGB
        processed_images.append(x)
    
    return np.stack(processed_images)

class FaceDetector:
    def __init__(self, model_path, num_gpus=1):
        print('Building network...')
        self.net = build_net('test', num_classes=2, model='dark')
        self.net.eval()
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location='cpu')
        self.net.load_state_dict(state_dict)
        
        # 多GPU并行
        self.num_gpus = num_gpus
        if torch.cuda.is_available():
            if num_gpus > 1:
                print(f"Using {num_gpus} GPUs for parallel processing")
                self.net = DataParallel(self.net, device_ids=list(range(num_gpus)))
            self.net = self.net.cuda()
        
        print('Model loaded successfully')
    
    def detect_batch(self, images, shrink=1.0):
        """批量检测人脸"""
        if not images:
            return []
        
        # 预处理
        processed_imgs = preprocess_image(images, shrink)
        tensor_imgs = torch.from_numpy(processed_imgs)
        
        # 移动到GPU
        if torch.cuda.is_available():
            tensor_imgs = tensor_imgs.cuda()
        
        # 推理
        with torch.no_grad():
            # 使用混合精度
            with torch.cuda.amp.autocast():
                outputs = self.net.test_forward(tensor_imgs)
        
        # 处理结果
        all_detections = []
        for i, img in enumerate(images):
            scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            detections = outputs[i].data.cpu().numpy() if self.num_gpus == 1 else outputs[i].data.cpu().numpy()
            
            boxes = []
            scores = []
            for j in range(detections.shape[0]):
                k = 0
                while k < detections.shape[1] and detections[j, k, 0] > 0.0:
                    pt = detections[j, k, 1:] * scale
                    score = detections[j, k, 0]
                    boxes.append([pt[0], pt[1], pt[2], pt[3]])
                    scores.append(score)
                    k += 1
            
            if not boxes:
                all_detections.append(np.array([[0, 0, 0, 0, 0.001]]))
            else:
                det_conf = np.array(scores)
                boxes = np.array(boxes)
                det_xmin = boxes[:, 0]
                det_ymin = boxes[:, 1]
                det_xmax = boxes[:, 2]
                det_ymax = boxes[:, 3]
                det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))
                all_detections.append(det)
        
        return all_detections

    def multi_scale_detect(self, image, max_im_shrink):
        """优化版多尺度检测"""
        detections = []
        scales = []
        
        # 确定测试尺度
        if max_im_shrink > 2.0:
            scales.extend([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
            if max_im_shrink > 2.0:
                scales.append(max_im_shrink)
        else:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, max_im_shrink]
        
        # 批量处理所有尺度
        scale_images = [cv2.resize(image, None, None, fx=s, fy=s, 
                                  interpolation=cv2.INTER_LINEAR) for s in scales]
        
        # 批量检测
        batch_detections = self.detect_batch(scale_images)
        
        # 处理结果
        for det, scale in zip(batch_detections, scales):
            if scale != 1.0:
                det[:, :4] /= scale
            detections.append(det)
        
        # 翻转测试
        flip_image = cv2.flip(image, 1)
        flip_det = self.detect_batch([flip_image])[0]
        if flip_det.shape[0] > 0:
            flip_det[:, 0] = image.shape[1] - flip_det[:, 2]
            flip_det[:, 2] = image.shape[1] - flip_det[:, 0]
            detections.append(flip_det)
        
        # 合并结果
        det = np.vstack(detections) if detections else np.array([])
        return bbox_vote(det) if det.size > 0 else np.array([[0,0,0,0,0.001]])

def bbox_vote(det):
    if det.size == 0:
        return np.array([[0,0,0,0,0.001]])
    
    order = det[:, 4].argsort()[::-1]
    det = det[order]
    dets = np.zeros((0, 5))
    
    while det.shape[0] > 0:
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (area[0] + area - inter)
        
        merge_index = np.where(iou >= 0.3)[0]
        if merge_index.shape[0] <= 1:
            det = np.delete(det, 0, 0)
            continue
            
        merge_det = det[merge_index]
        det = np.delete(det, merge_index, 0)
        
        weighted_det = np.zeros((1, 5))
        weighted_det[:, :4] = np.sum(merge_det[:, :4] * merge_det[:, 4:5], axis=0) / np.sum(merge_det[:, 4])
        weighted_det[:, 4] = np.max(merge_det[:, 4])
        dets = np.vstack([dets, weighted_det])
    
    return dets[:750]

def draw_boxes(image, detections, save_path):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    
    for det in detections:
        xmin, ymin, xmax, ymax, score = det
        if score > 0.8:
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, edgecolor='red', linewidth=1.5)
            plt.gca().add_patch(rect)
            plt.text(xmin, ymin - 2, f'{score:.2f}', 
                    color='red', fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, pad=1))
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)

    area_inter = inter_width * inter_height
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return area_inter / (area1 + area2 - area_inter + 1e-10)

def compute_mAP(detections_path, ground_truth_path, iou_threshold=0.5):
    ground_truths = {}
    for gt_file in glob.glob(os.path.join(ground_truth_path, '*.txt')):
        image_id = os.path.splitext(os.path.basename(gt_file))[0]
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                xmin, ymin, xmax, ymax = map(float, parts[:4])
                boxes.append([xmin, ymin, xmax, ymax])
            except:
                continue
        
        ground_truths[image_id] = boxes

    detections = {}
    for det_file in glob.glob(os.path.join(detections_path, '*.txt')):
        image_id = os.path.splitext(os.path.basename(det_file))[0]
        with open(det_file, 'r') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xmin, ymin, xmax, ymax, score = map(float, parts[:5])
                boxes.append([xmin, ymin, xmax, ymax, score])
            except:
                continue
        
        boxes.sort(key=lambda x: x[4], reverse=True)
        detections[image_id] = boxes

    aps = []
    for image_id, gt_boxes in ground_truths.items():
        det_boxes = detections.get(image_id, [])
        if not gt_boxes or not det_boxes:
            continue
        
        tp = np.zeros(len(det_boxes))
        fp = np.zeros(len(det_boxes))
        matched = [False] * len(gt_boxes)
        
        for i, det in enumerate(det_boxes):
            iou_max = 0
            best_match = -1
            
            for j, gt in enumerate(gt_boxes):
                if matched[j]:
                    continue
                
                iou = calculate_iou(det[:4], gt)
                if iou > iou_max:
                    iou_max = iou
                    best_match = j
            
            if iou_max >= iou_threshold:
                tp[i] = 1
                matched[best_match] = True
            else:
                fp[i] = 1
        
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        recall = tp_cumsum / len(gt_boxes)
        precision = tp_cumsum / (fp_cumsum + tp_cumsum + 1e-10)
        
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            precisions = precision[recall >= t]
            ap += np.max(precisions) / 11 if precisions.size > 0 else 0
        
        aps.append(ap)
    
    return np.mean(aps) if aps else 0

def main():
    # 配置参数
    config = {
        'model_path': '../../model/forDAINet/dark/dsfd.pth',
        'image_dir': '../../dataset/DarkFace/image',
        'save_dir': './result',
        'num_gpus': torch.cuda.device_count(),
        'batch_size': 2,  # 根据GPU内存调整
        'num_workers': 16   # 数据加载线程数
    }
    
    # 创建保存目录
    os.makedirs(os.path.join(config['save_dir'], 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(config['save_dir'], 'images'), exist_ok=True)
    
    # 加载图像
    img_paths = glob.glob(os.path.join(config['image_dir'], '*.png'))
    print(f'Found {len(img_paths)} images for processing')
    
    # 初始化检测器
    detector = FaceDetector(config['model_path'], num_gpus=config['num_gpus'])
    
    # 创建数据集和数据加载器
    dataset = FaceDetectionDataset(img_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 处理图像
    total_time = 0
    for batch_images, batch_names in tqdm(dataloader, desc='Processing images'):
        start_time = time.time()
        
        # 处理批次
        for i, (image, img_name) in enumerate(zip(batch_images, batch_names)):
            # 计算最大尺寸缩放
            max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5
            max_im_shrink = min(max_im_shrink, 3.0)
            
            # 检测人脸
            dets = detector.multi_scale_detect(image.numpy(), max_im_shrink)
            
            # 保存结果
            base_name = Path(img_name).stem
            annotation_path = os.path.join(config['save_dir'], 'annotations', f'{base_name}.txt')
            with open(annotation_path, 'w') as f:
                for det in dets:
                    f.write(f'{det[0]} {det[1]} {det[2]} {det[3]} {det[4]}\n')
            
            # 保存可视化结果
            image_path = os.path.join(config['save_dir'], 'images', f'{base_name}.png')
            draw_boxes(image.numpy(), dets, image_path)
        
        batch_time = time.time() - start_time
        total_time += batch_time
        avg_time = batch_time / len(batch_images)
        print(f'Batch processed in {batch_time:.2f}s | Avg per image: {avg_time:.2f}s')
    
    # 计算mAP
    print(f'Total processing time: {total_time:.2f}s')
    print(f'Average time per image: {total_time/len(img_paths):.2f}s')
    
    gt_path = '../../dataset/DarkFace/label'
    det_path = os.path.join(config['save_dir'], 'annotations')
    
    if os.path.exists(gt_path):
        mAP = compute_mAP(det_path, gt_path)
        print(f'mAP@0.5: {mAP:.4f}')
    else:
        print('Ground truth path not found, skipping mAP calculation')

if __name__ == '__main__':
    main()