import sys, os
import numpy as np
from natsort import natsorted
import torch
from PIL import Image

sys.path.insert(0, '/workspace/TwoStream-GeneSegNet/Code')
import metrics
import models as model_loader

test_dir = "/workspace/TwoStream-GeneSegNet/Dataset/test"
# Let's find the best baseline model path
model_dir = "/workspace/TwoStream-GeneSegNet"
import glob
files = glob.glob(model_dir + "/*")
model_path = [f for f in files if "baseline" in f][0]

print(f"Baseline TEST with model: {model_path}")

def load_images_with_heatmap(folder):
    images = []
    names = []
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('.')]
    subfolders = natsorted(subfolders)
    
    for sf in subfolders:
        img_path = os.path.join(folder, sf, 'images')
        hm_path = os.path.join(folder, sf, 'HeatMaps', 'HeatMap_all')
        
        if os.path.exists(img_path) and os.path.exists(hm_path):
            dapi_files = natsorted([f for f in os.listdir(img_path) if f.endswith('_image.jpg')])
            
            for im in dapi_files:
                dapi = np.array(Image.open(os.path.join(img_path, im)))
                if dapi.ndim == 3:
                    dapi = np.mean(dapi, axis=-1).astype(np.float32)
                else:
                    dapi = dapi.astype(np.float32)
                
                hm_name = im.replace('_image.jpg', '_gaumap_all.jpg')
                hm = np.array(Image.open(os.path.join(hm_path, hm_name)))
                if hm.ndim == 3:
                    hm = np.mean(hm, axis=-1).astype(np.float32)
                else:
                    hm = hm.astype(np.float32)
                
                img_stacked = np.stack([dapi, hm], axis=-1)
                images.append(img_stacked)
                names.append(im)
    return images, names

def load_labels(folder):
    labels = []
    names = []
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('.')]
    subfolders = natsorted(subfolders)
    for sf in subfolders:
        label_path = os.path.join(folder, sf, 'labels')
        if os.path.exists(label_path):
            lbls = natsorted([f for f in os.listdir(label_path) if f.endswith('.png')])
            for lb in lbls:
                label = np.array(Image.open(os.path.join(label_path, lb)))
                labels.append(label)
                names.append(lb)
    return labels, names

images, img_names = load_images_with_heatmap(test_dir)
labels, lbl_names = load_labels(test_dir)

device, gpu = model_loader.assign_device(use_torch=True, gpu=True)

model = model_loader.GeneSegModel(gpu=gpu, device=device,
                          pretrained_model=model_path,
                          diam_mean=34.,
                          nchan=2)

masks_true = []
masks_pred = []

for i, (image, label) in enumerate(zip(images, labels)):
    if i >= 420: break
    if label.max() == 0: continue
    
    out = model.eval(image, batch_size=4)
    masks, flows = out[:2]
    if isinstance(masks, list):
        masks = masks[0]
        
    masks_true.append(label)
    masks_pred.append(masks.astype(np.int32))

    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{len(images)}")

if len(masks_true) > 0:
    ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred)
    
    tp_sum = np.sum(tp, axis=0)
    fp_sum = np.sum(fp, axis=0)
    fn_sum = np.sum(fn, axis=0)
    
    precision = tp_sum / (tp_sum + fp_sum + 1e-7)
    recall = tp_sum / (tp_sum + fn_sum + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    print(f"\nBaseline Epoch 210 Results (on val set):")
    print(f"  AP@0.5: {np.mean(ap[:,0]):.3f}")
    print(f"  AP@0.75: {np.mean(ap[:,1]):.3f}")
    print(f"  AP@0.9: {np.mean(ap[:,2]):.3f}")
    
    print(f"\n  --- Detailed Metrics ---")
    print(f"  Threshold @0.5:")
    print(f"    Precision: {precision[0]:.3f}")
    print(f"    Recall:    {recall[0]:.3f}")
    print(f"    F1-Score:  {f1[0]:.3f}")
    
    print(f"  Threshold @0.75:")
    print(f"    Precision: {precision[1]:.3f}")
    print(f"    Recall:    {recall[1]:.3f}")
    print(f"    F1-Score:  {f1[1]:.3f}")
    
    print(f"  Threshold @0.9:")
    print(f"    Precision: {precision[2]:.3f}")
    print(f"    Recall:    {recall[2]:.3f}")
    print(f"    F1-Score:  {f1[2]:.3f}")
else:
    print("No valid masks!")
