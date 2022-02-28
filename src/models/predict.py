import pandas as pd
import os
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.callbacks import *

from annoy import AnnoyIndex


class Predict():
    def __init__(self):
        data_path = 'E:/Gokul/Research/Fashion Recommendations/Models'
        data_path = os.path.abspath(data_path)
        self.all_cnns = 'resnet-fashion.pth'
        self.rtr_inventory = AnnoyIndex(512*len(self.all_cnns))
        self.rtr_images = load_learner(os.path.abspath('./Models/'), self.all_cnns)

    # returns the embeddings for a single image,
    # from a single given CNN's last FC layer
    def get_embeddings_for_image(self, cnn, img_path):
        hook = hook_output(cnn.model[-1][-3])
        cnn.predict(open_image(img_path))
        hook.remove()
        return hook.stored.cpu()[0]

    # returns the concatenated embeddings for a single image,
    # from the given list of CNNs' last FC layer
    def get_combined_embeddings_for_image(self, img_path):
        embeddings = []
        for cnn in self.all_cnns:
            embeddings.append(self.get_embeddings_for_image(cnn, img_path))
        return np.concatenate(embeddings)

    # queries the given vector against the given ANN index
    def query_ann_index(self, embeddings, n=5):
        nns = self.rtr_inventory.get_nns_by_vector(
            embeddings, n=n, include_distances=True)
        img_paths = [self.rtr_images[i] for i in nns[0]]
        return img_paths, nns[1]

    # Get and display recs
    def get_recs(self, img_path, n=5):
        embedding = self.get_combined_embeddings_for_image(img_path)
        img_paths, sim_scores = self.query_ann_index(embedding, n)
        urls = [self.rtr_df.loc[img]['url'] for img in img_paths]
        return img_paths, sim_scores, urls
