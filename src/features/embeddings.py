from fastai.vision.data import ImageList
import pandas as pd
import numpy as np
import torch
import fastai
from fastai.vision import *
from matplotlib import pyplot as plt
from fastai.metrics import accuracy, top_k_accuracy

root_path = './input/'

def load_learner(data, pretrained_model, model_metrics, model_path):
    learner = cnn_learner(data, pretrained_model, metrics=model_metrics)
    learner.model = torch.nn.DataParallel(learner.model)
    learner = learner.load(model_path)
    return learner

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

if __name__ == '__main__':

	category_list = []
	image_path_list = []
	data_type_list = []

	# category names
	with open('./input/list_category_cloth.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			if i > 1:
				category_list.append(line.split(' ')[0])

	# category map
	with open('./input/list_category_img.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			if i > 1:
				image_path_list.append([word.strip() for word in line.split(' ') if len(word) > 0])


	# train, valid, test
	with open('./input/list_eval_partition.txt', 'r') as f:
		for i, line in enumerate(f.readlines()):
			if i > 1:
				data_type_list.append([word.strip() for word in line.split(' ') if len(word) > 0])

	data_df = pd.DataFrame(image_path_list, columns=['image_path', 'category_number'])
	data_df['category_number'] = data_df['category_number'].astype(int)
	data_df = data_df.merge(pd.DataFrame(data_type_list, columns=['image_path', 'dataset_type']), on='image_path')
	data_df['category'] = data_df['category_number'].apply(lambda x: category_list[int(x) - 1])
	#data_df = data_df.drop('category_number', axis=1)
	data_df.to_csv('./input/filenames.csv', index = False)

	image_list = ImageList.from_df(df=data_df, path=root_path, cols='image_path').split_none().label_from_df(cols = 'category')
	data = image_list.transform(size=224).databunch(bs=128).normalize(imagenet_stats)
	
	
	pretrained_model = models.resnet18
	model_path = 'resnet-fashion'
	model_metrics = [accuracy, partial(top_k_accuracy, k=1), partial(top_k_accuracy, k=5)]
	learner = load_learner(data, pretrained_model, model_metrics, model_path)

	saved_features = SaveFeatures(learner.model.module[1][4])
	_= learner.get_preds(data.train_ds)
	#_= learner.get_preds(DatasetType.Valid)

	img_path = [str(x) for x in (list(data.train_ds.items))]
	label = [data.classes[x] for x in (list(data.train_ds.y.items))]
	label_id = [x for x in (list(data.train_ds.y.items))]
	data_df_ouput = pd.DataFrame({'img_path': img_path, 'label': label, 'label_id': label_id})
	data_df_ouput['embeddings'] = np.array(saved_features.features).tolist()
	data_df_ouput.to_csv('./input/embeddings.csv', index = False)