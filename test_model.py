import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.LViT_new import LViTN
from utils import *
import cv2

def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = labs.astype(np.float32)
    tmp_3dunet = predict_save.astype(np.float32)

    # <<< ADD THIS >>>
    tmp_lbl = (tmp_lbl > 0.5).astype(np.uint8)
    tmp_3dunet = (tmp_3dunet > 0.5).astype(np.uint8)
    # <<< ADD THIS >>>

    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))

    cv2.imwrite(save_path, tmp_3dunet * 255)
    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    """
    Generate predictions and save visualization with metrics
    
    Args:
        model: Trained segmentation model
        input_img: Input image tensor
        text: Text embedding tensor
        img_RGB: RGB image (unused)
        labs: Ground truth labels
        vis_save_path: Base path for saving visualizations
        dice_pred: Accumulated Dice score (unused)
        dice_ens: Ensemble Dice score (unused)
        
    Returns:
        dice_pred_tmp: Dice score for this prediction
        iou_tmp: IoU score for this prediction
    """
    model.eval()

    # Forward pass through model
    output = model(input_img.cuda(), text.cuda())
    main_logits = output["out"]           # [1,1,56,56]
    probs = torch.sigmoid(main_logits)
    pred_class = (probs > 0.5).float()

    pred_np = pred_class[0,0].cpu().numpy()                                # [56,56]
    pred_up = cv2.resize(pred_np, (config.img_size, config.img_size))      # [224,224]

    predict_save = pred_up
    dice_pred_tmp, iou_tmp = show_image_with_dice(
            predict_save,
            labs,
            save_path=vis_save_path + '_predict' + model_type + '.jpg'
    )
    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    # Configure paths based on dataset
    if config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Covid19":
        test_num = 2113
        model_type = config.model_name
        model_path = "./Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    
    # Setup output directories
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    # Load trained model checkpoint
    checkpoint = torch.load(model_path, map_location='cuda')

    # Initialize model based on type
    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        model = LViTN(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=224, backbone_name='convnext_tiny', backbone_pretrained=True)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        model = LViTN(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=224, backbone_name='convnext_tiny', backbone_pretrained=True)

    else:
        raise TypeError('Please enter a valid name for the model type')

    # Setup model for inference
    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    
    # Prepare test dataset
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize metrics accumulators
    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    # Test loop with visualization
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # Extract data from batch
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            
            # Save ground truth label as image
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            
            # Configure figure size and remove margins
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()
            
            # Generate prediction and calculate metrics
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, test_text, None, lab,
                                                           vis_path + str(names),
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            
            # Accumulate metrics
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    
    # Print average metrics across test set
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)