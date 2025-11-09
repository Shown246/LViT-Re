# Train_model.py
# Clean training script with deep supervision support.
from getEm import init_embeddings
import os
import time
import random
import logging
import numpy as np
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.backends import cudnn

# Project imports - adapt if your package layout differs
import Config as config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, read_text, save_on_batch
from nets.LViT_new import LViTN
from Train_one_epoch import print_summary

torch.multiprocessing.set_start_method('spawn', force=True)
init_embeddings()
# Optional profiling
try:
    from thop import profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

# -------------------------
# Logger utilities
# -------------------------
def logger_config(log_path):
    """Configure logger writing to both file and console."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # clear existing handlers
    if logger.handlers:
        logger.handlers = []

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_checkpoint(state, save_dir):
    """Save checkpoint dict to disk. state should contain keys: epoch, best_model(bool), model (name)."""
    os.makedirs(save_dir, exist_ok=True)
    epoch = state.get('epoch', 0)
    best_model = state.get('best_model', False)
    model_name = state.get('model', 'model')
    if best_model:
        filename = os.path.join(save_dir, f'best_model-{model_name}.pth.tar')
    else:
        filename = os.path.join(save_dir, f'model-{model_name}-epoch{epoch:03d}.pth.tar')
    torch.save(state, filename)
    logger.info(f"Checkpoint saved to {filename}")


def worker_init_fn(worker_id):
    """Deterministic worker init for DataLoader."""
    seed = config.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# Loss / metrics helpers
# -------------------------
def _is_pred_dict(preds):
    return isinstance(preds, dict) and 'out' in preds


def compute_loss_from_preds(preds, masks, criterion, ds_weights=(0.2,0.3,0.4)):

    if masks.dim()==3:
        masks = masks.unsqueeze(1)

    # ------------------------------- IMPORTANT FIX
    # ensure main logits match target spatial size
    main_logits = preds['out'] if isinstance(preds, dict) else preds
    if main_logits.shape[-2:] != masks.shape[-2:]:
        masks = F.interpolate(masks, size=main_logits.shape[-2:], mode='bilinear', align_corners=False)
    # -------------------------------

    if not isinstance(preds, dict):
        loss = criterion(main_logits, masks)
        return loss, main_logits, {}

    ds_maps = preds.get('ds', [])

    loss_main = criterion(torch.sigmoid(main_logits), masks)
    total_loss = loss_main
    ds_losses = {}

    for idx, ds_map in enumerate(ds_maps):
        if ds_map is None: continue

        # match DS resolution to DS logits
        ds_target = F.interpolate(masks, size=ds_map.shape[-2:], mode='bilinear', align_corners=False)
        ds_loss = criterion(torch.sigmoid(ds_map), ds_target)
        w = ds_weights[idx] if idx<len(ds_weights) else 0.1
        total_loss = total_loss + w*ds_loss
        ds_losses[f"ds_{idx+1}"] = float(ds_loss.item())

    return total_loss, main_logits, ds_losses


def compute_dice_from_logits(logits, masks, eps=1e-6):
    if isinstance(logits, dict):
        logits = logits['out']

    # force shape: [B,1,H,W]
    if masks.dim() == 3:
        masks_t = masks.unsqueeze(1)
    else:
        masks_t = masks

    # ---- FIX SIZES ----
    if logits.shape[-2:] != masks_t.shape[-2:]:
        masks_t = masks_t.float()
        masks_t = F.interpolate(masks_t, size=logits.shape[-2:], mode='bilinear', align_corners=False)

    probs = torch.sigmoid(logits)

    B = probs.shape[0]
    probs_flat = probs.view(B, -1)
    masks_flat = masks_t.view(B, -1)

    inter = (probs_flat * masks_flat).sum(1)
    card = probs_flat.sum(1) + masks_flat.sum(1)

    return ((2.0 * inter + eps) / (card + eps)).mean().item()


def compute_iou_from_logits(logits, masks, thr=0.5, eps=1e-6):
    if isinstance(logits, dict):
        logits = logits['out']

    if masks.dim() == 3:
        masks_t = masks.unsqueeze(1)
    else:
        masks_t = masks

    # ---- FIX SIZES ----
    if logits.shape[-2:] != masks_t.shape[-2:]:
        masks_t = masks_t.float()
        masks_t = F.interpolate(masks_t, size=logits.shape[-2:], mode='bilinear', align_corners=False)

    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()

    B = probs.shape[0]
    preds_flat = preds.view(B, -1)
    masks_flat = masks_t.view(B, -1)

    inter = (preds_flat * masks_flat).sum(1)
    union = (preds_flat + masks_flat - preds_flat * masks_flat).sum(1)

    return ((inter + eps) / (union + eps)).mean().item()


# -------------------------
# Training / Validation loop functions
# -------------------------
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger,
                    ds_weights=(0.2, 0.3, 0.4), device=None):
    model.train()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0
    nb_batch = len(loader)
    start_time = time.time()
    end = start_time

    for i, (sampled_batch, names) in enumerate(loader, 1):
        images = sampled_batch['image'].to(device)
        masks = sampled_batch['label'].to(device)
        text = sampled_batch.get('text', None)
        if text is not None:
            # truncate or move to device
            if text.shape[1] > 10:
                text = text[:, :10, :].to(device)
            else:
                text = text.to(device)

        preds = model(images, text)

        loss, main_logits, ds_losses = compute_loss_from_preds(preds, masks.float(), criterion, ds_weights=ds_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics computed on main_logits only
        batch_dice = compute_dice_from_logits(main_logits.detach(), masks.detach())
        batch_iou = compute_iou_from_logits(main_logits.detach(), masks.detach())

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += batch_dice * bs
        total_iou += batch_iou * bs
        num_samples += bs

        batch_time = time.time() - end
        end = time.time()

        # Logging / print
        avg_loss = total_loss / num_samples
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
        lr_current = min(g['lr'] for g in optimizer.param_groups)

        if i % config.print_frequency == 0 or i == nb_batch:
            print_summary(epoch + 1, i, nb_batch, loss.item(), criterion.__class__.__name__, batch_time,
                          avg_loss, batch_time, batch_iou, avg_iou, batch_dice, avg_dice, 0, 0, 'Train', lr_current, logger)

        # Tensorboard logging
        if config.tensorboard and writer is not None:
            step = epoch * nb_batch + i
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Dice', batch_dice, step)
            writer.add_scalar('Train/IoU', batch_iou, step)
            # deep supervision losses
            for name, v in ds_losses.items():
                writer.add_scalar(f'Train/{name}', v, step)

    # optional scheduler step per epoch
    if lr_scheduler is not None:
        try:
            lr_scheduler.step()
        except Exception:
            pass

    avg_epoch_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_epoch_dice = total_dice / num_samples if num_samples > 0 else 0.0
    return avg_epoch_loss, avg_epoch_dice


def validate_one_epoch(loader, model, criterion, writer, epoch, logger, ds_weights=(0.2, 0.3, 0.4), device=None):
    model.eval()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = 0
    nb_batch = len(loader)
    end = time.time()

    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(loader, 1):
            images = sampled_batch['image'].to(device)
            masks = sampled_batch['label'].to(device)
            text = sampled_batch.get('text', None)
            if text is not None:
                if text.shape[1] > 10:
                    text = text[:, :10, :].to(device)
                else:
                    text = text.to(device)

            preds = model(images, text)
            loss, main_logits, ds_losses = compute_loss_from_preds(preds, masks.float(), criterion, ds_weights=ds_weights)

            batch_dice = compute_dice_from_logits(main_logits, masks)
            batch_iou = compute_iou_from_logits(main_logits, masks)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_dice += batch_dice * bs
            total_iou += batch_iou * bs
            num_samples += bs

            batch_time = time.time() - end
            end = time.time()

            if config.vis_frequency and (epoch % config.vis_frequency == 0):
                vis_path = os.path.join(config.visualize_path, str(epoch))
                os.makedirs(vis_path, exist_ok=True)
                try:
                    save_on_batch(images, masks, preds['out'], names, vis_path)   # <<< FIX HERE
                except Exception as e:
                    logger.warning(f"vis save failed: {e}")

            avg_loss = total_loss / num_samples
            avg_dice = total_dice / num_samples
            avg_iou = total_iou / num_samples
            if i % config.print_frequency == 0 or i == nb_batch:
                print_summary(epoch + 1, i, nb_batch, loss.item(), criterion.__class__.__name__, batch_time,
                              avg_loss, batch_time, batch_iou, avg_iou, batch_dice, avg_dice, 0, 0, 'Val', 0.0, logger)

            # Tensorboard
            if config.tensorboard and writer is not None:
                step = epoch * nb_batch + i
                writer.add_scalar('Val/Loss', loss.item(), step)
                writer.add_scalar('Val/Dice', batch_dice, step)
                writer.add_scalar('Val/IoU', batch_iou, step)
                for name, v in ds_losses.items():
                    writer.add_scalar(f'Val/{name}', v, step)

    avg_epoch_loss = total_loss / num_samples if num_samples > 0 else 0.0
    avg_epoch_dice = total_dice / num_samples if num_samples > 0 else 0.0
    return avg_epoch_loss, avg_epoch_dice


# -------------------------
# Main training loop
# -------------------------
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # set device and seeds
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Data transforms and datasets
    train_tf = RandomGenerator(output_size=[config.img_size, config.img_size])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

    # Load text and datasets depending on task
    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
    elif config.task_name == 'Covid19':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
    else:
        # Fallback: no text
        train_text = None
        val_text = None

    train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf, image_size=config.img_size)
    val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    # Model & config
    config_vit = config.get_CTranS_config()
    logger.info('Transformer heads: {}'.format(config_vit.transformer.num_heads))
    logger.info('Transformer layers: {}'.format(config_vit.transformer.num_layers))
    logger.info('Transformer expand ratio: {}'.format(config_vit.expand_ratio))

    model = LViTN(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=224, backbone_name='convnext_tiny', backbone_pretrained=True)
    model = model.to(device)

    # Profile model if requested
    if _HAS_THOP and config.profile_model:
        model.eval()
        inp = torch.randn(2, 3, config.img_size, config.img_size).to(device)
        txt = torch.randn(2, 10, 768).to(device)
        try:
            flops, params = profile(model, inputs=(inp, txt))
            logger.info(f"Model FLOPs: {flops}, Params: {params}")
        except Exception:
            logger.info("Profiling failed.")

    # DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1 and config.use_dataparallel:
        logger.info(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # Loss, optimizer, scheduler
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    lr_scheduler = None
    if config.cosineLR:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)

    # TensorBoard writer
    writer = None
    if tensorboard and config.tensorboard:
        log_dir = config.tensorboard_folder
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs at {log_dir}")

    # training loop
    best_dice = -1.0
    best_epoch = 0
    start_epoch = 0

    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\n===== Epoch {epoch + 1}/{config.epochs} =====")
        # Train
        train_loss, train_dice = train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, logger,
                                                 ds_weights=config.ds_weights, device=device)
        # Validate
        val_loss, val_dice = validate_one_epoch(val_loader, model, criterion, writer, epoch, logger,
                                                ds_weights=config.ds_weights, device=device)

        logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} Dice {train_dice:.4f} | Val Loss {val_loss:.4f} Dice {val_dice:.4f}")

        # Scheduler step with metric if applicable
        if lr_scheduler is not None:
            try:
                lr_scheduler.step(epoch + 1)
            except Exception:
                try:
                    lr_scheduler.step(val_loss)
                except Exception:
                    pass

        # Save best model after burn-in epochs
        if val_dice > best_dice:
            if epoch + 1 > config.save_after_epoch:
                logger.info(f"Validation Dice improved {best_dice:.4f} -> {val_dice:.4f}; saving model.")
                best_dice = val_dice
                best_epoch = epoch + 1
                ckpt = {
                    'epoch': epoch + 1,
                    'best_model': True,
                    'model': config.model_name,
                    'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_checkpoint(ckpt, config.model_path)
        else:
            logger.info(f"No improvement (best {best_dice:.4f} at epoch {best_epoch})")

        # early stopping check
        early_stop_count = epoch - best_epoch + 1
        logger.info(f"Early stopping counter: {early_stop_count}/{config.early_stopping_patience}")
        if early_stop_count > config.early_stopping_patience:
            logger.info("Early stopping triggered.")
            break

    if writer:
        writer.close()

    logger.info(f"Training finished. Best val dice: {best_dice:.4f} at epoch {best_epoch}")
    return model


# -------------------------
# Entrypoint
# -------------------------
if __name__ == '__main__':
    # configure logger
    logger = logger_config(config.logger_path)

    # reproducibility
    if config.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # create save dir
    os.makedirs(config.save_path, exist_ok=True)

    # run
    model = main_loop(model_type=config.model_name, tensorboard=True)
