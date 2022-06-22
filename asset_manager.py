# Because our original output from COCOMaskExtract
# can not directly feed into lama engine
# This file does reindexing and renaming

from select import select
import hydra
from omegaconf import OmegaConf
import os
import shutil
import glob
import pickle
import tqdm
import torch
from PIL import Image
import numpy as np

if torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')

def select_masks(masks: list, pkl_object: dict):
    return [masks[0]]

def read_mask(path: str):
    mask = Image.open(path)
    mask = np.array(mask)
    mask = torch.tensor(mask, dtype=torch.uint8).to(device)
    return mask

@hydra.main(config_path='.', config_name='config_asset_manager.yaml')
def main(config: OmegaConf):
    indir = config['indir']
    outdir = config['outdir']

    # Delete the output path
    shutil.rmtree(outdir, ignore_errors=True)
    if (not os.path.isdir(outdir)): os.mkdir(outdir)

    entries = glob.glob(indir + '/*')
    for index, entry in tqdm.tqdm(enumerate(entries)):
        mask_paths = glob.glob(entry + '/*.png')
        img_path = glob.glob(entry + '/*.jpg')[0]
        with open(glob.glob(entry + '/*.pkl')[0], 'rb') as f:
            pkl_object = pickle.load(f)
        
        masks = [read_mask(mask_path) for mask_path in mask_paths]
        masks = select_masks(masks, pkl_object)

        # Merge masks
        if len(masks) == 0: continue
        output_mask = masks[0]
        for mask in masks:
            output_mask = torch.logical_or(output_mask, mask).to(torch.uint8)

        output_mask = 255 * output_mask
        png_mask = Image.fromarray(output_mask.cpu().numpy(), mode='RGB')

        # copy to destination
        png_mask.save(outdir + f'/{index}_mask.png')
        shutil.copyfile(img_path, f'{outdir}/{index}.jpg')

if __name__ == '__main__':
    main()