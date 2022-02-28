import io
from dash_html_components.Button import Button
import matplotlib
matplotlib.use('Agg')
import dash
import time
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html_comp
import base64
from fastai.vision.image import open_image, pil2tensor
import matplotlib
from pandas.core import base
import torch
from glob import glob
import os
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from annoy import AnnoyIndex
import torch
import fastai
from fastai.vision import *
import shutil
from fastai.metrics import accuracy, top_k_accuracy
import ast
import matplotlib.pyplot as plt
import webbrowser
from threading import Timer
import base64
from io import BytesIO as _BytesIO

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

root_path = './input/'
data_df = pd.read_csv('./input/filenames.csv')

ntree = 512
annoy_tree = AnnoyIndex(ntree, 'angular')
annoy_tree.load('./input/annoy_tree.ann')

pretrained_model = models.resnet18
model_metrics = [accuracy, partial(top_k_accuracy, k=1), partial(top_k_accuracy, k=5)]
model_path = 'resnet-fashion'

image_list = ImageList.from_df(df=data_df, path=root_path, cols='image_path').split_none().label_from_df(cols = 'category')
data = image_list.transform(size=224).databunch(bs=128).normalize(imagenet_stats)


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

app.layout = html_comp.Div(children=[
    html_comp.Div(children = 
        html_comp.H1(
                    children='OUTFIT RECOMMENDATIONS',
                    style = {
                            'textAlign' : 'center'
                            }
                    ),
                className ='col-8',
                style = {
                        'padding-top' : '1%'
                        }
                ),
    html_comp.Div(children = 
        html_comp.H4(
                    children=html_comp.B('Upload input image: '),
                    style = {
                            'textAlign' : 'left'
                            },
                    ),
                className='col-8',
                style = {'padding-top' : '1%'}    
                ),
    dcc.Upload(id='upload-image',
        children = html_comp.Button('Upload File'),
        # Allow multiple files to be uploaded
        multiple=True
            ),
    html_comp.Div(id = 'output-image-upload')
            ])

def parse_contents(contents, recommendations):
    return html_comp.Div(children = [
    html_comp.Div(children = [ 
        html_comp.H4(
                    children=html_comp.B('Uploaded image: '),
                    style = {
                            'textAlign' : 'left'
                            },
                    ),
        html_comp.Img(id = 'uploaded-image', src = contents)
                            ],
                className = 'col-8',
                style = {'padding-top' : '1%'}
                ),
    
    html_comp.Div(children = [
        html_comp.H4(
                    children=html_comp.B('Recommendations: '),
                    style = {
                            'textAlign' : 'left'
                            },
                    ),
        html_comp.Img(id = 'recommendations', src = recommendations)
                            ],
                className = 'col-8',
                style = {'padding-top' : '1%'}
                ),
                ])

def recommendation(image):
    
    img = b64_to_pil(image[0].split("base64,")[-1])
    img = fastai.vision.Image(pil2tensor(img.convert('RGB'), np.float32).div_(255)).resize(224)
    
    learner = load_learner(data, pretrained_model, model_metrics, model_path)
    
    saved_features = SaveFeatures(learner.model.module[1][4])
    _= learner.predict(img)

    embedding = saved_features.features
    
    similar_images_df = get_similar_images_annoy_centroid(annoy_tree, embedding[0], 30)
    
    print(similar_images_df[similar_images_df.duplicated()])
    
    output_data = show_similar_images(similar_images_df, learner, fig_size = (15, 15))
    return "data:image/png;base64,{}".format(output_data)
    

def open_browser():
    webbrowser.get("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s").open("http://127.0.0.1:8050/")            

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))

def update_output(contents):
    if contents is None:
        raise dash.exceptions.PreventUpdate()
    children = [
        parse_contents(contents[0], recommendation(contents))
    ]
    return children

if __name__ == '__main__':
    Timer(15, open_browser).start();
    app.run_server(debug = True)
	#recommendation('./input/img/2-in-1_Space_Dye_Athletic_Tank/img_00000001.jpg')
	