# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    # Format the summary string for logging
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr) # Add learning rate for training
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time) # Add average time
    summary += string
    logger.info(summary) # Log the summary string


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    # Determine if the model is in training or validation mode based on its state
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time() # Start timing the epoch
    time_sum, loss_sum = 0, 0 # Initialize accumulators for time and loss
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0 # Initialize accumulators for metrics
    dices = [] # List to store dice scores for potential further analysis

    # Iterate over the data loader
    for i, (sampled_batch, names) in enumerate(loader, 1):

        # Get the name of the loss function/criterion
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Extract images, masks, and text from the batch
        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        # Truncate text sequence if it's too long (hardcoded limit of 10)
        if text.shape[1] > 10:
            text = text[ :, :10, :]
        
        # Move data to GPU
        images, masks, text = images.cuda(), masks.cuda(), text.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        # Forward pass: get model predictions
        preds = model(images, text)
        # Calculate the loss
        out_loss = criterion(preds, masks.float())  # Loss
        # print(model.training)


        if model.training: # Only perform backward pass and optimizer step in training mode
            optimizer.zero_grad() # Clear gradients
            out_loss.backward() # Backpropagate the loss
            optimizer.step() # Update model parameters

        # Calculate metrics for the current batch
        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)

        # Calculate time taken for this batch
        batch_time = time.time() - end
        
        # Save visualizations if it's validation and the epoch is at the specified frequency
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        
        # Store the dice score for this batch
        dices.append(train_dice)

        # Accumulate time, loss, and metrics weighted by batch size
        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        # Calculate running averages for time, loss, and metrics
        # Handle the last batch which might have a different size
        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else: # Calculate averages based on full batch sizes processed so far
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time() # Reset timer for next batch
        torch.cuda.empty_cache() # Clear GPU cache periodically

        # Print summary statistics periodically based on print frequency
        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        # Log metrics to TensorBoard if enabled
        if config.tensorboard:
            step = epoch * len(loader) + i # Global step for TensorBoard
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache() # Clear GPU cache again

    # Step the learning rate scheduler at the end of the epoch (if provided)
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Return the average loss and average dice score for the epoch
    return average_loss, train_dice_avg