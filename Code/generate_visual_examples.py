import sys, os
sys.path.insert(0, '/workspace/TwoStream-GeneSegNet/Code')
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob
from natsort import natsorted
import models as model_loader
from models import TwoStreamGeneSegModel

test_dir = '/workspace/TwoStream-GeneSegNet/Dataset/test'

# Load models
device, gpu = model_loader.assign_device(use_torch=True, gpu=True)

# Baseline
bl_model_dir = '/workspace/TwoStream-GeneSegNet'
bl_model_path = [f for f in glob.glob(bl_model_dir + '/*') if 'epoch_211' in f and 'GeneSegNet' in f][0]
bl_model = model_loader.GeneSegModel(gpu=gpu, device=device, pretrained_model=bl_model_path, diam_mean=34., nchan=2)

# TwoStream
ts_model_dir = '/workspace/TwoStream-GeneSegNet'
ts_model_path = [f for f in glob.glob(ts_model_dir + '/*') if 'epoch_141' in f and 'twostream_geneseg' in f][0]
ts_model = TwoStreamGeneSegModel(gpu=gpu, device=device, pretrained_model=ts_model_path, diam_mean=34., nchan=2, n_genes=1, cross_attn_layers=(4,))

# Get an image
folder = test_dir
sf = 'DAPI_7-3_left'
img_path = os.path.join(folder, sf, 'images')
hm_path = os.path.join(folder, sf, 'HeatMaps', 'HeatMap_all')
label_path = os.path.join(folder, sf, 'labels')

# Let's take a few specific images that might show a good comparison
images_to_test = [0, 4] # Index 0 and 4 

im_files = natsorted([f for f in os.listdir(img_path) if f.endswith('_image.jpg')])
lb_files = natsorted([f for f in os.listdir(label_path) if f.endswith('.png')])

for idx in images_to_test:
    im = im_files[idx]
    lb = lb_files[idx]
    
    dapi = np.array(Image.open(os.path.join(img_path, im)))
    if dapi.ndim == 3: dapi = np.mean(dapi, axis=-1).astype(np.float32)
    else: dapi = dapi.astype(np.float32)
    
    hm_name = im.replace('_image.jpg', '_gaumap_all.jpg')
    hm = np.array(Image.open(os.path.join(hm_path, hm_name)))
    if hm.ndim == 3: hm = np.mean(hm, axis=-1).astype(np.float32)
    else: hm = hm.astype(np.float32)
    
    image = np.stack([dapi, hm], axis=-1)
    label = np.array(Image.open(os.path.join(label_path, lb)))
    
    # Predict
    out_bl = bl_model.eval(image, batch_size=4)[0]
    if isinstance(out_bl, list): out_bl = out_bl[0]
    
    out_ts = ts_model.eval(image, batch_size=4)[0]
    if isinstance(out_ts, list): out_ts = out_ts[0]
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    axes[0].imshow(dapi, cmap='gray')
    axes[0].set_title('1) DAPI', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(hm, cmap='hot')
    axes[1].set_title('2) RNA Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(label, cmap='tab20')
    axes[2].set_title('3) Ground Truth', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(out_bl, cmap='tab20')
    axes[3].set_title('4) Baseline', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    axes[4].imshow(out_ts, cmap='tab20')
    axes[4].set_title('5) Two-Encoder', fontsize=12, fontweight='bold')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/workspace/TwoStream-GeneSegNet/Results/segmentation_sample_{idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
print("Görsel karşılaştırmalar kaydedildi: /workspace/TwoStream-GeneSegNet/Results/segmentation_sample_X.png")

