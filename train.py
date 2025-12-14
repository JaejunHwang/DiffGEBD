import argparse
import os
import pickle
import time
from collections import defaultdict
from contextlib import suppress
from datetime import timedelta
import random
import numpy as np
import torch
import torch.distributed as dist
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from datasets import build_dataloader
from modeling import cfg, build_model
from solver.solver_utils import build_optimizer, build_scheduler
from utils.distribute import synchronize, all_gather, is_main_process
from utils.eval import eval_f1
from utils.misc import SmoothedValue, MetricLogger
import wandb

def make_inputs(inputs, device):
    if isinstance(inputs, list):
        return [img.to(device) for img in inputs]
    else:
        return inputs.to(device)



def make_targets(cfg, inputs, device):
    targets = inputs['labels'].to(device)
    return targets


def train_one_epoch(cfg, args, model, device, optimizer, scheduler, data_loader, summary_writer, auto_cast, loss_scaler, epoch):
    model.train()

    start = time.time()
    for i, inputs in enumerate(data_loader):
        
        samples = make_inputs(inputs['imgs'], device)
        targets = inputs['labels'].to(device)
        masks = None

        model_start = time.time()
        with auto_cast():
            loss_dict = model(samples, targets, masks)
            total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        if cfg.SOLVER.AMPE:
            loss_scaler.scale(total_loss).backward()

            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            scheduler.step()
        else:
            total_loss.backward()
            if cfg.SOLVER.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD)
            optimizer.step()
            scheduler.step()

        if is_main_process():
            summary_writer.global_step += 1

            if len(loss_dict) > 1:
                summary_writer.update(**loss_dict)

            summary_writer.update(lr=optimizer.param_groups[0]['lr'], total_loss=total_loss,
                                  total_time=time.time() - start, model_time=time.time() - model_start)
            start = time.time()

            speed = summary_writer.total_time.avg
            eta = str(timedelta(seconds=int((len(data_loader) - i - 1) * speed)))
            if i % 10 == 0:
                print('Epoch{:02d} ({:04d}/{:04d}): {}, Eta:{}'.format(epoch,
                                                                       i,
                                                                       len(data_loader),
                                                                       str(summary_writer),
                                                                       eta
                                                                       ), flush=True)
                

@torch.no_grad()
def validate(cfg, args, model, device, data_loader, epoch):
    if cfg.INPUT.END_TO_END:
        return validate_end_to_end(cfg, args, model, device, data_loader, epoch)

    model_pred_dict = {}
    model.eval()
    start_time = time.time()
    num_frames = 0
    for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
        samples = make_inputs(inputs, device)
        num_frames += samples['imgs'].shape[0]

        vids = inputs['vid']
        frame_idxs = inputs['frame_idx']
        scores = model(samples)

        scores = scores.cpu().numpy()
        for vid, frame_idx, score in zip(vids, frame_idxs, scores):
            if vid not in model_pred_dict.keys():
                model_pred_dict[vid] = {}
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []
            model_pred_dict[vid]['frame_idx'].append(frame_idx)
            model_pred_dict[vid]['scores'].append(score)

    synchronize()
    metrics = {}
    data_list = all_gather(model_pred_dict)
    if not is_main_process():
        metrics['F1'] = 0.00
        return metrics
    total_time = time.time() - start_time
    print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, len(data_loader.dataset), total_time / len(data_loader.dataset)))
    print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

    model_pred_dict = defaultdict(dict)
    for p in data_list:
        for vid in p:
            if 'frame_idx' not in model_pred_dict[vid]:
                model_pred_dict[vid]['frame_idx'] = []
                model_pred_dict[vid]['scores'] = []

            model_pred_dict[vid]['frame_idx'].extend(p[vid]['frame_idx'])
            model_pred_dict[vid]['scores'].extend(p[vid]['scores'])

    for vid in model_pred_dict:
        frame_idx = np.array(model_pred_dict[vid]['frame_idx'])
        scores = np.array(model_pred_dict[vid]['scores'])
        _, indices = np.unique(frame_idx, return_index=True)
        frame_idx = frame_idx[indices]
        scores = scores[indices]

        indices = np.argsort(frame_idx)
        model_pred_dict[vid]['frame_idx'] = frame_idx[indices].tolist()
        model_pred_dict[vid]['scores'] = scores[indices].tolist()

    gt_path = f'data/k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl'
    f1, rec, prec = eval_f1(model_pred_dict, gt_path, threshold=cfg.TEST.THRESHOLD, protocol=cfg.TEST.PROTOCOL)
    print('F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))
    metrics['F1'] = f1
    metrics['Rec'] = rec
    metrics['Prec'] = prec
    return metrics


@torch.no_grad()
def validate_end_to_end(cfg, args, model, device, data_loader, epoch):
    metrics = {}
    if cfg.TEST.PRED_FILE:
        if not is_main_process():
            return metrics, metrics
        with open(cfg.TEST.PRED_FILE, 'rb') as f:
            model_pred_dict = pickle.load(f)
        print(f'Load results from {cfg.TEST.PRED_FILE}.')
    else:
        model_pred_dict = defaultdict(dict)
        model.eval()

        start_time = time.time()
        num_frames = 0
        for i, inputs in enumerate(tqdm(data_loader, total=len(data_loader))):
            samples = make_inputs(inputs['imgs'], device)
            targets = inputs['labels'].to(device)
            
            num_frames += (targets.shape[0] * targets.shape[1])

            outputs = model(samples, targets, sampling_timesteps=cfg.DIFFUSION.SAMPLING_TIMESTEPS)  # (b, t)
            for batch_idx, frame_indices in enumerate(inputs['frame_indices']):
                vid = inputs['vid'][batch_idx]
                scores = outputs[batch_idx]
                if 'frame_masks' in inputs:
                    frame_mask = inputs['frame_masks'][batch_idx]
                    frame_indices = frame_indices[frame_mask]
                    scores = scores[frame_mask]

                model_pred_dict[vid]['frame_idx'] = frame_indices.tolist()
                model_pred_dict[vid]['scores'] = scores.tolist()


        synchronize()
        data_list = all_gather(model_pred_dict)
        if not is_main_process():
            metrics['F1'] = 0.00
            return metrics, metrics
        total_time = time.time() - start_time
        print('Cost {:.2f}s for evaluating {} videos, {:.4f}s/video'.format(total_time, len(data_loader.dataset), total_time / len(data_loader.dataset)))
        print('{:.9f}ms/frame'.format(total_time * 1000 / num_frames))

        model_pred_dict = {}
        for p in data_list:
            model_pred_dict.update(p)

    if not cfg.TEST.PRED_FILE:
        save_path = os.path.join(args.output_dir, 'model_pred_dict_ep{}.pkl'.format(epoch))
        with open(save_path, 'wb') as f:
            pickle.dump(model_pred_dict, f)
        print(f'Saved results to {save_path}.')

    if data_loader.dataset.name == 'GEBD':
        gt_path = os.path.join('data', f'k400_mr345_{data_loader.dataset.split}_min_change_duration0.3.pkl')
    elif data_loader.dataset.name == 'TAPOS':
        gt_path = os.path.join('data', f'TAPOS_for_GEBD_{data_loader.dataset.split}.pkl')
    else:
        raise NotImplemented

    if args.all_thres:
        rel_dis_thres = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        rel_dis_thres = [0.05]

    results, pred_dict, gt_dict = eval_f1(model_pred_dict, gt_path,
                                                        threshold=cfg.TEST.THRESHOLD,
                                                        return_pred_dict=True,
                                                        rel_dis_thres=rel_dis_thres,
                                                        protocol=cfg.TEST.PROTOCOL)
    list_rec = []
    list_prec = []
    list_f1 = []
    
    for th in rel_dis_thres:
        f1, rec, prec = results[th]
        list_rec.append(rec)
        list_prec.append(prec)
        list_f1.append(f1)

    headers = rel_dis_thres + ['Avg']

    avg_rec = np.mean(list_rec)
    avg_prec = np.mean(list_prec)
    avg_F1 = np.mean(list_f1)
    
    
    tabulate_data = [
        ['Recall'] + list_rec + [avg_rec],
        ['Precision'] + list_prec + [avg_prec],
        ['F1'] + list_f1 + [avg_F1],
    ]
    print(f'Results for {data_loader.dataset.name}_{data_loader.dataset.split}:')
    print(tabulate(tabulate_data, headers=headers, floatfmt='.4f'))

    f1, rec, prec = results[0.05]
    print('___F1@0.05: {:.4f}, Rec: {:.4f}, Prec: {:.4f}'.format(f1, rec, prec))

    return results


def main(cfg, args):
    train_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TRAIN, train=True)
    val_data_loader = build_dataloader(cfg, args, cfg.DATASETS.TEST, train=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(cfg)
    model = model.to(device)

    if is_main_process() and not args.test_only:
        print(model)

    start_epoch = -1
    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)
        start_epoch = state_dict['epoch']
        if is_main_process():
            print('Loaded from {}, Epoch: {}'.format(args.resume, start_epoch), flush=True)

    exp_name = '_ann{}_dim{}_len{}'.format(cfg.INPUT.ANNOTATORS,
                                            cfg.MODEL.DIMENSION,
                                            cfg.INPUT.SEQUENCE_LENGTH,   
                                            )
    output_dir = cfg.OUTPUT_DIR + exp_name
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    with open(os.path.join(output_dir,'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        torch.multiprocessing.set_start_method('spawn')
    
    if args.test_only:
        results = validate(cfg, args, model, device, val_data_loader, epoch=-1)
        return
    
    optimizer_config = {}
    optimizer_config['warmup'] = cfg.SOLVER.WARMUP
    optimizer_config['epochs'] = cfg.SOLVER.MAX_EPOCHS
    optimizer_config['warmup_epochs'] = cfg.SOLVER.WARMUP_EPOCHS
    optimizer_config['schedule_type'] = cfg.SOLVER.SCHEDULER
    optimizer_config["schedule_steps"] = cfg.SOLVER.MILESTONES
    optimizer_config["gamma"] = cfg.SOLVER.GAMMA
    optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
    scheduler = build_scheduler(optimizer, optimizer_config, len(train_data_loader))
     
    if args.resume:
        for name, obj in [('optimizer', optimizer), ('scheduler', scheduler)]:
            if name in state_dict:
                obj.load_state_dict(state_dict[name])
                if is_main_process():
                    print('Loaded {} from {}'.format(name, args.resume), flush=True)

    summary_writer = MetricLogger(log_dir=os.path.join(output_dir, 'logs')) if is_main_process() else None
    if summary_writer is not None:
        summary_writer.add_meter('lr', SmoothedValue(fmt='{value:.5f}'))
        summary_writer.add_meter('total_time', SmoothedValue(fmt='{avg:.3f}s'))
        summary_writer.add_meter('model_time', SmoothedValue(fmt='{avg:.3f}s'))

    auto_cast = torch.cuda.amp.autocast if cfg.SOLVER.AMPE else suppress
    loss_scaler = torch.cuda.amp.GradScaler() if cfg.SOLVER.AMPE else None

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    best_f1 = 0.
    for epoch in range(start_epoch + 1, cfg.SOLVER.MAX_EPOCHS):
        
        train_one_epoch(cfg, args, model, device, optimizer, scheduler, train_data_loader, summary_writer, auto_cast, loss_scaler, epoch)

        if len(cfg.DIFFUSION.VALIDATION_TIMESTEPS) == 1 or cfg.DIFFUSION.DETERMINISTIC:
            results = validate(cfg, args, model, device, val_data_loader, epoch)

            if is_main_process():
                if args.wandb:
                    wandb.log({'Val/F1@0.05' : results[0.05][0],
                            'Val/Rec@0.05' : results[0.05][1],
                            'Val/Prec@0.05' : results[0.05][2],}, step=epoch)
        
        else:
            for sampling_steps in cfg.DIFFUSION.VALIDATION_TIMESTEPS:
                cfg.DIFFUSION.SAMPLING_TIMESTEPS = sampling_steps
                results = validate(cfg, args, model, device, val_data_loader, epoch)
                if is_main_process():
                    if args.wandb:
                        wandb.log({f'Val_st{sampling_steps}/F1@0.05' : results[0.05][0],
                            f'Val_st{sampling_steps}/Rec@0.05' : results[0.05][1],
                            f'Val_st{sampling_steps}/Prec@0.05' : results[0.05][2],}, step=epoch)
                        
        if is_main_process():
            if args.wandb:
                wandb.log({'Train/loss' : summary_writer.meters['total_loss'].global_avg,
                        'Train/lr' : summary_writer.meters['lr'].value
                    }, step=epoch)
                    
            model_state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            
            if results[0.05][0] >= best_f1:
                best_f1 = results[0.05][0]
                save_path = os.path.join(output_dir, f'model_best.pth')
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)    
                print(f'Saved best model at {epoch}')

            if results[0.05][0] >= 0.73:
                save_path = os.path.join(output_dir, f'model_epoch{epoch:02d}.pth')
                torch.save({
                    'model': model_state_dict,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_path)
            with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
                f.write('Epoch: {:02d},Rel@0.05 F1: {:.4f}, Rec: {:.4f}, Prec: {:.4f}\n'.format(epoch, results[0.05][0], results[0.05][1], results[0.05][2]))
            print('Saved to {}'.format(save_path))
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--test-only", action='store_true')
    parser.add_argument("--train-test", action='store_true')
    parser.add_argument("--all-thres", action='store_true', help='test using all thresholds [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]')
    parser.add_argument('--gebd_data_dir', default='your_kinetics_folder', type=str)
    parser.add_argument('--tapos_data_dir', default='your_tapos_folder', type=str)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--proj_name", type=str, default='your_proj')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cuda_deterministic', action='store_true', default=False)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.local_rank:
        args.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if torch.cuda.is_available():
        if args.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if is_main_process():
        print('Args: \n{}'.format(args))
        print('Configs: \n{}'.format(cfg))

        if args.wandb:
            exp_name = cfg.OUTPUT_DIR.replace('output/', '') + '_' + args.exp_name
            run = wandb.init(entity='user_wandb', 
                            project=args.proj_name,
                            name=exp_name)
    
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    np.random.seed(seed)
    random.seed(seed) 
    if args.cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(cfg, args)
