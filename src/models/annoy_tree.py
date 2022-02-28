from annoy import AnnoyIndex
import pandas as pd
import ast
from tqdm import tqdm

if __name__ == '__main__':
	data_df_output = pd.read_csv('./input/embeddings.csv')
	ntree = 100
	metric_choice = 'angular'
	annoy_tree = AnnoyIndex(len(ast.literal_eval(data_df_output['embeddings'][0])), metric = metric_choice)

	with tqdm(total = len(data_df_output['embeddings']), desc = 'Images') as pbar:
		for i, vector in enumerate(data_df_output['embeddings']):
			annoy_tree.add_item(i, ast.literal_eval(vector))
			pbar.update()
	annoy_tree.build(ntree)
	annoy_tree.save('./input/annoy_tree.ann')