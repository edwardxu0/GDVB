#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import random
import skimage


random.seed(0)
np.random.seed(0)
NB_PROPS = 10
IMG_ROOT = { "driving":"/p/d4v/dls2fc/udacity-driving/data",
             "imagenet":"/p/d4v/dls2fc"
}
IMG_FOLDERS = { "driving": ["center"],
                "imagenet": ["imagenet.224","imagenet.112","imagenet.56","imagenet.28","imagenet.14",]
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Property choosing and normalization.")
    parser.add_argument("dataset", type=str, choices=["driving","imagenet"])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset = args.dataset

    img_root = IMG_ROOT[dataset]

    if dataset == "":
        pass
    elif dataset == "imagenet":
        out_root = dataset+str(NB_PROPS)
        if not os.path.exists(out_root):
            os.mkdir(out_root)
        
        img_classes =  os.listdir(os.path.join(img_root, IMG_FOLDERS[dataset][0], 'img_val'))
        random.shuffle(img_classes)
        img_classes = img_classes[:NB_PROPS]
        true_label_dict = {l.split(' ')[0]:i for i, l in enumerate(open("synset.txt",'r').readlines())}
        
        img_file_names = []
        for img_f in img_classes:
            img_files = os.listdir(os.path.join(img_root, IMG_FOLDERS[dataset][0], 'img_val', img_f))
            random.shuffle(img_files)
            img = img_files[0]
            img_file_names += [img]

        for sub_folder in IMG_FOLDERS[dataset]:
            if not os.path.exists(os.path.join(out_root, sub_folder)):
                os.mkdir(os.path.join(out_root, sub_folder))

            for i in range(NB_PROPS):
                in_path = os.path.join(img_root, sub_folder, "img_val", img_classes[i], img_file_names[i])
                true_label = true_label_dict[img_classes[i]]
                out_path = os.path.join(out_root, sub_folder, str(true_label) +'_'+ img_file_names[i])
                img = skimage.io.imread(in_path).astype(float)
                img = img[[2,1,0]]
                img[0] = img[0] - 103.939
                img[1] = img[1] - 116.779
                img[2] = img[2] - 123.68
                np.save(out_path, img)
