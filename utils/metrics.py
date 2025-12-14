import numpy as np  
import torch
import torch.nn as nn
import pickle


def do_eval(gt_dict, pred_dict, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0

    for vid_id in list(gt_dict.keys()):

        # filter by avg_f1 score
        if gt_dict[vid_id]['f1_consis_avg'] < 0.3:
            continue

        if vid_id not in pred_dict.keys():
            num_pos_all += len(gt_dict[vid_id]['substages_myframeidx'][0])
            continue

        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['num_frames'] - 1  # number of frames

        # remove detected boundary outside the action instance
        tmp = []
        if type(bdy_timestamps_det) == int:
            print('y')
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            num_pos_all += len(gt_dict[vid_id]['substages_myframeidx'][0])
            continue
        num_det = len(bdy_timestamps_det)
        num_det_all += num_det

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_myframeidx']
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1 = 0
    else:
        f1 = 2 * rec * prec / (rec + prec)

    return f1, rec, prec


def get_idx_from_score_by_threshold(threshold=0.5, nms_ths=0.1, seq_indices=None, seq_scores=None):
    if len(seq_scores) == 1:
        seq_scores = seq_scores[0]
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    bdy_indices = []
    internals_indices = []
    '''
    seq_score: 0~seq_len 
    seq_indices: seq_len 길이의 실제 해당하는 Frame 위치
    특정 threshold 넘는 frame 유지 후, frame < threshold 를 기준으로 구획 나눔
    해당 구획 (Interval) 에서는 center 만 boundary 로 기준 (nms)
    '''
    
    for i in range(len(seq_scores)):
        if seq_scores[i] >= threshold:
            internals_indices.append(i)
        elif seq_scores[i] < threshold and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)
            internals_indices = []

        if i == len(seq_scores) - 1 and len(internals_indices) != 0:
            bdy_indices.append(internals_indices)

    bdy_indices_in_video = []
    if len(bdy_indices) != 0:
        for internals in bdy_indices:
            center = round(np.mean(internals))
            bdy_indices_in_video.append(seq_indices[center])
    
    return bdy_indices_in_video

def get_idx_from_score_by_threshold_nms(threshold=0.3, kernel_size=11, seq_indices=None, seq_scores=None):
    seq_len = len(seq_indices)
    seq_indices = np.array(seq_indices)
    seq_scores = np.array(seq_scores)
    
    ''' 
    Non-max suppression
    '''
    seq_tensor = torch.tensor(seq_scores).unsqueeze(0)

    # for GEBD 100-frame, kernel_size = 11
    # for GEBD 50-frame, kernel_size = 5
    # for TAPOS 50-frame, kernel_size = 5
    maxpool = nn.MaxPool1d(kernel_size, 1, kernel_size//2)
    peak = (seq_tensor == maxpool(seq_tensor))
    peak[seq_tensor < threshold] = False
    nonzero = torch.nonzero(peak.squeeze()).squeeze().tolist()
    bdy_indices_in_video = seq_indices[nonzero]
    
    return bdy_indices_in_video

def make_pred_dict(gt_dict, my_pred, threshold=0.5, rel_dis_thres=0.05, nms_mode=False, kernel_size=7):
    

    pred_dict = dict()
    for vid in my_pred:
        if vid in gt_dict:
            if not nms_mode:
                det_t = np.array(get_idx_from_score_by_threshold(threshold=threshold,
                                                             seq_indices=my_pred[vid]['frame_idx'],
                                                             seq_scores=my_pred[vid]['scores']))
            
            else:
                det_t = np.array(get_idx_from_score_by_threshold_nms(threshold=threshold,
                                                             kernel_size=kernel_size,
                                                             seq_indices=my_pred[vid]['frame_idx'],
                                                             seq_scores=my_pred[vid]['scores']))
            

            assert np.all(det_t >= 0)
            if det_t.size == 1:
                pred_dict[vid] = [det_t.item()]
            else:
                pred_dict[vid] = det_t.tolist()
            

    return pred_dict

def f1_maxpred_vid(vid_id, gt_dict, pred_dict_list, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)

    f1_all = 0
    for pred_dict in pred_dict_list:
    
        if vid_id not in pred_dict.keys():
            f1_all += 0
            continue
    
        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['num_frames'] - 1  # number of frames

        # remove detected boundary outside the action instance
        tmp = []
        if type(bdy_timestamps_det) == int:
            print('y')
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            f1_all += 0
            continue
        num_det = len(bdy_timestamps_det)

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_myframeidx']
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        # GTs 와 하나의 Pred 에 대한 평균
        ann_best = np.argmax(f1_tmplist)
        f1_all += f1_tmplist[ann_best]
    
    f1_all /= len(pred_dict_list)
    return f1_all

def f1_maxgt_vid(vid_id, gt_dict, pred_dict_list, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)

    f1_all = 0
    gt_num = len(gt_dict[vid_id]['substages_myframeidx'])
    f1_gt_dict = {i: [] for i in range(gt_num)}
    for pred_dict in pred_dict_list:
    
        if vid_id not in pred_dict.keys():
            f1_all += 0
            continue
    
        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['num_frames'] - 1  # number of frames

        # remove detected boundary outside the action instance
        tmp = []
        if type(bdy_timestamps_det) == int:
            print('y')
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            f1_all += 0
            continue
        num_det = len(bdy_timestamps_det)

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_myframeidx']
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        
        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            if bdy_timestamps_list_gt_allraters[ann_idx] == []:
                continue

            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1
            f1_gt_dict[ann_idx].append(f1)
        # GTs 와 하나의 Pred 에 대한 평균
    
    for f1_idx in range(len(f1_gt_dict)):
        if f1_gt_dict[f1_idx] == []:
            gt_num -= 1
            continue
        f1_all += np.max(f1_gt_dict[f1_idx])
    
    gt_num = len(f1_gt_dict)
    f1_all /= gt_num

    return f1_all   

def f1_vid_pred_gt(vid_id, gt_dict, pred_dict_list, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)

    distance_all = 0
    for pred_dict in pred_dict_list:
    
        if vid_id not in pred_dict.keys():
            distance_all += 1
            continue
    
        # detected timestamps
        bdy_timestamps_det = pred_dict[vid_id]

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['num_frames'] - 1  # number of frames

        # remove detected boundary outside the action instance
        tmp = []
        if type(bdy_timestamps_det) == int:
            print('y')
        for det in bdy_timestamps_det:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_timestamps_det = tmp
        if bdy_timestamps_det == []:
            distance_all += 1
            continue
        num_det = len(bdy_timestamps_det)

        # compare bdy_timestamps_det vs. each rater's annotation, pick the one leading the best f1 score
        bdy_timestamps_list_gt_allraters = gt_dict[vid_id]['substages_myframeidx']
        f1_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_timestamps_list_gt_allraters))

        for ann_idx in range(len(bdy_timestamps_list_gt_allraters)):
            bdy_timestamps_list_gt = bdy_timestamps_list_gt_allraters[ann_idx]
            num_pos = len(bdy_timestamps_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_list_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_list_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        # GTs 와 하나의 Pred 에 대한 평균
        distance = 1 - np.mean(f1_tmplist)
        distance_all += distance
    
    distance_all /= len(pred_dict_list)

    return distance_all

def f1_vid_gt_gt(vid_id, gt_dict, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)

    distance_all = 0

    gt_num = len(gt_dict[vid_id]['substages_myframeidx'])

    for i in range(gt_num):
        for j in range(gt_num):
        # for j in range(i+1, gt_num):
        # detected timestamps
            ins_start = 0
            ins_end = gt_dict[vid_id]['num_frames'] - 1

            bdy_timestamps_det = gt_dict[vid_id]['substages_myframeidx'][i]
            bdy_timestamps_gt = gt_dict[vid_id]['substages_myframeidx'][j]
    
            # remove detected boundary outside the action instance
            tmp = []
            if type(bdy_timestamps_det) == int:
                print('y')
            for det in bdy_timestamps_det:
                tmpdet = det + ins_start
                if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                    tmp.append(tmpdet)
            bdy_timestamps_det = tmp
            if bdy_timestamps_det == []:
                distance_all += 1
                continue
            num_det = len(bdy_timestamps_det)

            # remove detected boundary outside the action instance: GT 로 사용하는 부분
            tmp = []
            if type(bdy_timestamps_gt) == int:
                print('y')
            for gt in bdy_timestamps_gt:
                tmpgt = gt + ins_start
                if tmpgt >= (ins_start) and tmpgt <= (ins_end):
                    tmp.append(tmpgt)
            bdy_timestamps_gt = tmp

            num_pos = len(bdy_timestamps_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)

            distance = 1 - f1
            distance_all += distance
    
    assert(gt_num != 1), vid_id
    # denom = gt_num * (gt_num -1) / 2
    denom = gt_num ** 2
    return distance_all / denom


def f1_vid_pred_pred(vid_id, gt_dict, pred_dict_list, threshold=0.05):
    # recall precision f1 for threshold 0.05(5%)

    distance_all = 0
    pred_num = len(pred_dict_list)
    
    for pred_dict in pred_dict_list:
        if vid_id not in pred_dict.keys():
            return 1

    for i in range(pred_num):
        for j in range(pred_num):
        # for j in range(i+1, pred_num):
            ins_start = 0
            ins_end = gt_dict[vid_id]['num_frames'] - 1  # number of frames

            bdy_timestamps_det = pred_dict_list[i][vid_id]
            bdy_timestamps_gt = pred_dict_list[j][vid_id]
    
            # remove detected boundary outside the action instance
            tmp = []
            if type(bdy_timestamps_det) == int:
                print('y')
            for det in bdy_timestamps_det:
                tmpdet = det + ins_start
                if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                    tmp.append(tmpdet)
            bdy_timestamps_det = tmp
            if bdy_timestamps_det == []:
                distance_all += 1
                continue
            num_det = len(bdy_timestamps_det)

            # remove detected boundary outside the action instance: GT 로 사용하는 부분
            tmp = []
            if type(bdy_timestamps_gt) == int:
                print('y')
            for gt in bdy_timestamps_gt:
                tmpgt = gt + ins_start
                if tmpgt >= (ins_start) and tmpgt <= (ins_end):
                    tmp.append(tmpgt)
            bdy_timestamps_gt = tmp

            num_pos = len(bdy_timestamps_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_timestamps_gt), len(bdy_timestamps_det)))
            for ann1_idx in range(len(bdy_timestamps_gt)):
                for ann2_idx in range(len(bdy_timestamps_det)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_timestamps_gt[ann1_idx] - bdy_timestamps_det[ann2_idx])
            for ann1_idx in range(len(bdy_timestamps_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= threshold * (ins_end - ins_start + 1):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)

            distance = 1 - f1
            distance_all += distance
    
    # denom = pred_num * (pred_num - 1) / 2
    denom = pred_num ** 2
    return distance_all / denom


def eged(vids, gt_dict, pred_dict_list):
    except_cnt = 0
    d1s = 0
    d2s = 0
    d3s = 0
    
    for vid in vids:
        if gt_dict[vid]['f1_consis_avg'] < 0.3:
            except_cnt += 1
            continue

        d1 = 0
        d2 = 0 
        d3 = 0

        d1 = f1_vid_pred_gt(vid, gt_dict, pred_dict_list)
        d2 = f1_vid_pred_pred(vid, gt_dict, pred_dict_list)
        d3 = f1_vid_gt_gt(vid, gt_dict)

        d1s += d1
        d2s += d2
        d3s += d3

    d1s /= (len(vids) - except_cnt)
    d2s /= (len(vids) - except_cnt)
    d3s /= (len(vids) - except_cnt)


    ged = 2*d1s - d2s - d3s
    return ged, d2s


def das(vids, gt_dict, pred_dict_list):
    f1_max = 0
    div_aln = 0

    for vid in vids:
        f1 = f1_maxgt_vid(vid, gt_dict, pred_dict_list)
        f1_max += f1

        div_gt = f1_vid_gt_gt(vid, gt_dict)
        div_pred = f1_vid_pred_pred(vid, gt_dict, pred_dict_list)
        div_aln += 1 - np.abs(div_gt - div_pred)
    
    f1_max /= len(vids)
    div_aln /= len(vids)

    harmonic = harmonic_mean(f1_max, div_aln)
    arithmetic = arithmetic_mean(f1_max, div_aln)

    return f1_max, div_aln, harmonic, arithmetic

def get_f1_sym(vids, gt_dict, pred_dict_list, threshold=0.05):
    f1_maxhar = 0
    f1_maxgt_all = 0
    f1_predgt_all = 0
    except_cnt = 0

    for vid in vids:
        if gt_dict[vid]['f1_consis_avg'] < 0.3:
            except_cnt += 1
            continue

        f1_maxgt = f1_maxgt_vid(vid, gt_dict, pred_dict_list, threshold)
        f1_maxgt_all += f1_maxgt
        f1_predgt = f1_maxpred_vid(vid, gt_dict, pred_dict_list, threshold)
        f1_predgt_all += f1_predgt
        f1_maxhar += harmonic_mean(f1_maxgt, f1_predgt)

    f1_sym = f1_maxhar
    f1_g2p = f1_maxgt_all
    f1_p2g = f1_predgt_all
    
    f1_sym /= (len(vids) - except_cnt)
    f1_g2p /= (len(vids) - except_cnt)
    f1_p2g /= (len(vids) - except_cnt)

    # f1_sym, f1_g2p, f1_p2g
    return f1_sym, f1_g2p, f1_p2g

def harmonic_mean(a, b):
    return 2 * a * b / (a + b + 1e-10)

def arithmetic_mean(a, b):
    return (a + b) / 2