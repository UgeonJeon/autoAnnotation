import os
import cv2
from glob import glob 
import os.path as osp
import numpy as np

import magic
import re
import pandas as pd
import gc

from tqdm.notebook import tqdm
import torch

from ocsort.ocsort import OCSort


def verify_annotation(annotation):
    if type(annotation) != np.ndarray:
        annotation = np.asarray(annotation)
    
    x1, x2 = annotation[:,1] - annotation[:,3]/2, annotation[:,1] + annotation[:,3]/2
    y1, y2 = annotation[:,2] - annotation[:,4]/2, annotation[:,2] + annotation[:,4]/2

    old = np.stack((annotation[:,0], x1, x2, y1, y2), axis = 1)
    wrong_idx = (old[:,1] > 1) | (old[:,2] < 0) | (old[:,3] > 1) | (old[:,4] < 0)

    if sum(wrong_idx) == 0: # 모든 라벨링 정상
        annotation[:,1:]  = np.trunc(annotation[:,1:]*1e4)/1e4
        return 'flawless', annotation
    elif sum(wrong_idx) == len(annotation): # 모든 라벨링 비정상
        return 'error', None
    else:
        new = old[~wrong_idx]
        new[:,1::2] = np.where(new[:,1::2] < 0, 0.0001, new[:,1::2])
        new[:,2::2] = np.where(new[:,2::2] > 1, 0.9999, new[:,2::2])

        result_annotation = np.zeros(len(new)*5).reshape(len(new), 5)
        result_annotation[:,0] = new[:,0]
        result_annotation[:,1] = (new[:,1] + new[:,2])/2
        result_annotation[:,2] = (new[:,3] + new[:,4])/2
        result_annotation[:,3] = (new[:,2] - new[:,1])
        result_annotation[:,4] = (new[:,4] - new[:,3])
        result_annotation[:,1:] = np.trunc(result_annotation[:,1:]*1e4)/1e4
        return 'revised', result_annotation


def xywh2xyxy(x, image_shape = False):
    """
    image_shape = (width, height)
    """
    if not isinstance(x, (np.ndarray)):
        x = np.array(x)

    # Convert nx4 boxes from [cx, cy, w, h, cls] to [x1, y1, x2, y2, cls] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
    if image_shape:
        y[:,:4] = y[:,:4] * [int(image_shape[0]), int(image_shape[1]),int(image_shape[0]), int(image_shape[1])]
    return y


def xyxy2xywh(x, image_shape = False):
    """
    image_shape = (width, height)
    """
    if not isinstance(x, (np.ndarray)):
        x = np.array(x)

    # Convert nx4 boxes from [x1, y1, x2, y2, cls] to [cx, cy, w, h, cls]
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y
    y[:, 2] = (x[:, 2] - x[:, 0])         # width x
    y[:, 3] = (x[:, 3] - x[:, 1])        # width y

    if image_shape:    
        y[:,:4] = y[:,:4] / [int(image_shape[0]), int(image_shape[1]),int(image_shape[0]), int(image_shape[1])]
    return y


def IoU( box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def calculateZOOM(img1, img2, scale_factor = 0.7):
    small1 = cv2.resize(img1, None, fx=scale_factor, fy=scale_factor)
    small2 = cv2.resize(img2, None, fx=scale_factor, fy=scale_factor)

    # Convert to grayscale
    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute the ORB descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Match the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches in the order of their distances
    matches = sorted(matches, key = lambda x : x.distance)

    # Calculate the zoom level
    if len(matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        zoom = np.abs(np.linalg.det(M))
    return zoom


class autoAnnotate:
    
    def __init__(self, dataset_path, recursive):
        self.dataset_path = dataset_path
        self.recursive = recursive

    def prepare_Detection_df(self):
        """
        caution! Only Available for jpg images
        dets = [cx, cy, w, h, 1(conf), cls]
        dets_xyxy = [x1, y1, x2, y2, 1(conf), cls]
        """
        images = sorted(glob(self.dataset_path + '**/*.jpg', recursive = self.recursive))
        img_path_dict = {osp.basename(f) : f for f in images}
        annote_dict = dict()
        annote_xyxy_dict = dict()
        image_shape = list(map(int, re.findall('(\d+)x(\d+)', magic.from_file(images[0]))[-1]))

        for image in images:
            img_name = image.split('/')[-1]
            try :
                with open(image[:-4]+ '.txt', 'r') as f:
                    temp = np.asarray(list(map(lambda x: list(np.float64(x.strip().split( )))+ [1], f.readlines())))
                    annotation = temp[:,1:]
                    annotation = np.append(annotation, temp[:,0]).reshape(1,6)

                    annotation_xyxy = xywh2xyxy(annotation, image_shape)
                    annote_dict[img_name] = annotation
                    annote_xyxy_dict[img_name]= annotation_xyxy
            except :
                annote_xyxy_dict[img_name]= np.nan
                annote_dict[img_name] = np.nan

        df = pd.DataFrame(columns = ['image', 'path','dets', 'dets_xyxy', 'tracker_forward', 'tracker_backward', 'tracker_rst', 'interpolate_rst', 'final_pred'])  
        df['image'] = img_path_dict.keys()
        df['path'] = df['image'].apply(lambda x : img_path_dict[x])
        df['dets'] = df['image'].apply(lambda x : annote_dict[x])
        df['dets_xyxy'] = df['image'].apply(lambda x : annote_xyxy_dict[x])
        print(f"found {len(df)} images / shape : {image_shape[0]}x{image_shape[1]}")
        return df, image_shape


    def get_properDets(self, prev_dets, new_dets, tracking_valid_iou):
        """
        prev_dets, new_dets 라벨링1:1 매칭해 prev_dets와 tracking_valid_iou 이상인 new_dets만 남김
        """
        valid_idx = []
        for idx, prev_det in enumerate(prev_dets):
            for new_det in new_dets:
                iou = IoU(new_det, prev_det)
                if iou >= tracking_valid_iou:
                    valid_idx.append(idx)
        valid_idx = list(set(valid_idx))
        valid_prev_dets = prev_dets[valid_idx]
        return valid_prev_dets

    
    # predict with tracker
    def create_ocSort_tracker(self):
        ocsort_forward = OCSort(det_thresh = 0.1, iou_threshold = 0.1, min_hits = 1, delta_t = 7,)
        ocsort_backward = OCSort(det_thresh = 0.1, iou_threshold = 0.1, min_hits = 1, delta_t = 7,)
        return ocsort_forward, ocsort_backward


    def predictDet(self, tracker, df, direction, tracking_valid_iou, pred_count, image_shape):
        """
        direction : forward, backward
        """
        if direction  == 'forward':
            out_df = df.sort_values(by = 'image', ascending = True)
        elif direction  == 'backward':
            out_df = df.sort_values(by = 'image', ascending = False)

        predictable_state = False
        prev_Detected = False
        prev_img = None
        prev_TrueDets = None
        prev_TrueDets_xywh = None
        zoom_factor = []
        predDet_count = 1
        interpolate_det_count = 0

        for idx in tqdm(out_df.index):
            image_name = out_df._get_value(idx, 'image')
            det = out_df._get_value(idx, 'dets_xyxy')
            img_path = out_df._get_value(idx, 'path')
            img = cv2.imread(img_path)

            if isinstance(det, (np.ndarray)) :
                if prev_Detected == False:
                    prev_Detected = True
                    predictable_state = False
                elif prev_Detected:
                    predictable_state = True

                prev_img = img
                prev_TrueDets = det
                prev_TrueDets_xywh = xyxy2xywh(det, image_shape)
                predDet_count = 1
                zoom_factor = []
                tracker.update(det,img)
            else :
                det = np.empty((0, 6))
                pred_dets = tracker.update(det,img, return_predict = True)
                pred_dets = self.get_properDets(pred_dets, prev_TrueDets, tracking_valid_iou)

                if predictable_state and predDet_count < pred_count + 1 and len(pred_dets) >=1:
                    zoom = calculateZOOM(prev_img, img)
                    zoom_factor.append(zoom)
                    if zoom >= 0.6 and zoom <= 1.6:
                        pred_dets_xywh = xyxy2xywh(pred_dets, image_shape)
                        pred_dets_xywh[:,2] = prev_TrueDets_xywh[:,2].max() * np.mean(zoom_factor)
                        pred_dets_xywh[:,3] = prev_TrueDets_xywh[:,3].max() * np.mean(zoom_factor)
                        out_df._set_value(idx, f'tracker_{direction}', pred_dets_xywh)        
                        predDet_count += 1
                        interpolate_det_count += 1
                    else:
                        predictable_state = False
                        zoom_factor = []

                prev_img = img
                prev_Detected = False

        if direction  == 'backward':
            out_df = out_df.sort_values(by = 'image', ascending=True)
        return out_df


    def update_predDet(self, rst_df, tracking_valid_iou):
        out_df = rst_df.copy()
        part_images = out_df.loc[(out_df['tracker_forward'].notna())|(out_df['tracker_backward'].notna())].index
        for idx in part_images:
            pred1 = out_df._get_value(idx, 'tracker_forward')
            pred2 = out_df._get_value(idx, 'tracker_backward')

            if type(pred1) == float:
                out_df._set_value(idx, 'tracker_rst', pred2)
            elif type(pred2) == float:
                out_df._set_value(idx, 'tracker_rst', pred1)
            elif len(pred1) == 1 and len(pred2) == 1: 
                iou = IoU(pred1[0], pred2[0])
                if iou >= tracking_valid_iou:
                    new_pred = (pred1  + pred2)/2
                    out_df._set_value(idx, 'tracker_rst', new_pred)
                else:
                    continue
        out_df = out_df.sort_values(by = 'image', ascending=True)
        return out_df

    
    # interpolate
    def interpolate_indexList(self, df, interpolate_count):
        """
         : 어떤 column을 기준으로 interpolate할 것인지 지정
        2프레임 연속으로 det이 존재해야 interpolate 가능
        """
        part_df = df[['image','dets']].copy()

        empty_idx_list = list(df.loc[df['dets'].isna()].index)
        part_df['prev_dets'] = part_df['dets'].shift(1)
        part_df['next_dets'] = part_df['dets'].shift(-1)
        interpolatable_idx_list = list(part_df.loc[(part_df['dets'].notna())&((part_df['prev_dets'].notna())|(part_df['next_dets'].notna()))].index)

        idx = 0
        prev_i = False
        rtn_dict = dict()
        temp_list = []
        rtn_list = []

        for i in empty_idx_list:
            if not prev_i:
                temp_list.append(i)
                prev_i = i
                continue
            if i - prev_i == 1:
                temp_list.append(i)
                prev_i = i
            else:
                prev_i = i
                if len(temp_list) <= interpolate_count:
                    min_idx = min(temp_list) - 1
                    max_idx = max(temp_list) + 1
                    if min_idx in interpolatable_idx_list and max_idx in interpolatable_idx_list:
                        rtn_list.append(min_idx)
                        rtn_list.append(max_idx)
                temp_list = [i]
        del part_df
        return rtn_list

    
    def interpolateDet(self, df, interpolate_count):
        """
        return shape : (1, 5)
        interpolate detections with arithmetic sequence(등차수열)
            2프레임 연속으로 det이 존재해야 interpolate 가능
         : 어떤 column을 기준으로 interpolate할 것인지 지정
        interpolate_count : interpolate할 최대 연속된 빈칸; ex) 30프레임 이상 연속되게 비어있는 경우 interpolate하지 않음
        """
        out_df = df.copy()
        ipt_list = self.interpolate_indexList(out_df, interpolate_count)

        for idx in range(len(ipt_list) // 2):
            start_idx = ipt_list[2 * idx]
            end_idx = ipt_list[2 * idx + 1]

            start_det = out_df._get_value(start_idx,'dets')
            end_det = out_df._get_value(end_idx, 'dets')
            if len(start_det) == 1 and len(end_det) == 1:
                diff_det = (end_det - start_det) / (end_idx - start_idx)
                dcx, dcy, dw, dh =  diff_det[0][0], diff_det[0][1], diff_det[0][2], diff_det[0][3]

                for mv_idx, df_idx in enumerate(range(start_idx + 1, end_idx)):
                    new_cx = start_det[:,0] + dcx * (mv_idx + 1)
                    new_cy = start_det[:,1] + dcy * (mv_idx + 1)
                    new_w = start_det[:,2] + dw * (mv_idx + 1)
                    new_h = start_det[:,3] + dh * (mv_idx + 1)
                    new_det = f"{new_cx[0]} {new_cy[0]} {new_w[0]} {new_h[0]} 1 0"
                    new_det2 = np.array([list(map(lambda x : np.float64(x), new_det.split()))])
                    out_df._set_value(df_idx, 'interpolate_rst', new_det2)
        return out_df
    

    def merge_finalDet(self, rst_df, merge_valid_iou):
        out_df = rst_df.copy()
        part_images = out_df.loc[(out_df['tracker_rst'].notna())|(out_df['interpolate_rst'].notna())].index
        for idx in part_images:
            pred1 = out_df._get_value(idx, 'tracker_rst')
            pred2 = out_df._get_value(idx, 'interpolate_rst')

            if type(pred1) == float:
                out_df._set_value(idx, 'final_pred', pred2)
            elif type(pred2) == float:
                out_df._set_value(idx, 'final_pred', pred1)
            elif len(pred1) == 1 and len(pred2) == 1: 
                iou = IoU(pred1[0], pred2[0])
                if iou >= merge_valid_iou:
                    _, new_pred = verify_annotation(pred2)
                    out_df._set_value(idx, 'final_pred', new_pred)
                else:
                    new_pred = np.vstack((pred1  + pred2))
                    _, new_pred = verify_annotation(new_pred)
                    out_df._set_value(idx, 'final_pred', new_pred)

        print(f"{out_df['final_pred'].notna().sum()}/{out_df['dets'].isna().sum()} blank images auto-annotated")
        print(f"    tracker {out_df['tracker_rst'].notna().sum()} / interpolate {out_df['interpolate_rst'].notna().sum()}")
        out_df = out_df.sort_values(by = 'image', ascending=True)
        return out_df
    
    
    def save_newAnnotations(self, df, col_name, save_path):
        if osp.isdir(save_path) == False:
            os.mkdir(save_path)

        new_annotes = df.loc[df[col_name].notnull()].index
        for idx in new_annotes:
            imageName = df._get_value(idx, 'image')
            temp_new = df._get_value(idx, col_name)
            tmpe_new2 = temp_new[:,:4]
            tmpe_new2 = np.append(temp_new[:,-1], tmpe_new2).reshape(1,5)
            ret, new_preds = verify_annotation(tmpe_new2)

            if ret != 'error':
                with open(save_path + imageName[:-4] + '.txt', 'w') as f:
                    for pred_det in new_preds.tolist():
                        to_write = f"{int(pred_det[0])} {pred_det[1]} {pred_det[2]} {pred_det[3]} {pred_det[4]}\n"
                        f.write(to_write)


    def execute(self, tracking_valid_iou = 0.2, merge_valid_iou = 0.6, pred_count = 20, interpolate_count = 20):
        """
        tracking_valid_iou = tracker_forward와 tracker_backward를 비교할 IOU
        merge_valid_iou = tracker와 interpolate를 비교할 IOU
        pred_count = tracker을 이용해 predict할 프레임 수 
        interpolate_count = interpolate할 최대 연속된 빈 detection; ex) 30프레임 이상 연속되게 비어있는 경우 interpolate하지 않음
        """
        df, image_shape = self.prepare_Detection_df()
        ocsort_tracker_forward, ocsort_tracker_backward = self.create_ocSort_tracker()

        tacker_df1 = self.predictDet(ocsort_tracker_forward, df, direction='forward', pred_count = pred_count, 
                                    tracking_valid_iou = tracking_valid_iou, image_shape=image_shape)
        tacker_df2 = self.predictDet(ocsort_tracker_backward, tacker_df1, direction='backward', pred_count = pred_count,
                                    tracking_valid_iou = tracking_valid_iou, image_shape=image_shape)
        tracker_rst_df  = self.update_predDet(tacker_df2, tracking_valid_iou)
        
        interpolate_rst_df = self.interpolateDet(tracker_rst_df, interpolate_count)
        
        result_df = self.merge_finalDet(interpolate_rst_df, merge_valid_iou)
        
        return result_df