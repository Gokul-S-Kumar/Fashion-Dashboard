import pandas as pd
import fastai
from fastai.vision import *
import torch
from annoy import AnnoyIndex
import base64
from io import BytesIO as _BytesIO
import PIL
import numpy as np

data_df = pd.read_csv('./input/filenames.csv')

def get_similar_images_annoy_centroid(annoy_tree, vector_value, number_of_items=12):
    similar_img_ids = annoy_tree.get_nns_by_vector(vector_value, number_of_items+1)
    return data_df.iloc[similar_img_ids[1:]]

def load_learner(data, pretrained_model, model_metrics, model_path):
    learner = cnn_learner(data, pretrained_model, metrics=model_metrics)
    learner.model = torch.nn.DataParallel(learner.model)
    learner = learner.load(model_path)
    return learner

def show_similar_images(similar_images_df, learner, fig_size=[10,10], hide_labels=True):
    buf = io.BytesIO()
    if hide_labels:
        category_list = []
        for i in range(len(similar_images_df)):
            # replace category with blank so it wont show in display
            category_list.append(CategoryList(similar_images_df['category_number'].values*0, [''] * len(similar_images_df)).get(i))
    else:
        category_list = [learner.data.train_ds.y.reconstruct(y) for y in similar_images_df['category_number']]
    learner.data.show_xys([open_image('./input/' + img_id) for img_id in similar_images_df['image_path']],
                                category_list, figsize=fig_size)
    plt.savefig(buf, format = 'png')
    plt.close()
    return base64.b64encode(buf.getbuffer()).decode("utf8")

def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = PIL.Image.open(buffer)
    return im

class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()