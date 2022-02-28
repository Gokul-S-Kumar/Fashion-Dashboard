import matplotlib
matplotlib.use('Agg')
from app import app
from app import server
import dash_html_components as html_comp
import dash_core_components as dcc
import fastai
from fastai.vision import *
from annoy import AnnoyIndex
from fastai.metrics import accuracy, top_k_accuracy
from utils import *
import dash
from dash.dependencies import Input, Output

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

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = app.layout = html_comp.Div(children=[
    html_comp.Div(children = 
        html_comp.H2(
                    children='OUTFIT RECOMMENDATIONS',
                    style = {
                            'textAlign' : 'center',
							'background' : 'black',
							'color' : 'white'
                            }
                    ),
                className ='col-8',
                style = {
                        'padding-top' : '1%'
                        }
                ),
	dcc.Tabs(id = "all-tabs-inline", value = 'tab-1', children = 
		[
            dcc.Tab(label = 'IMAGE INPUT', value = 'tab-1', children = 
			[
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
        			multiple=True,
				),
				html_comp.Div(id = 'output-image-upload-tab1')
			],
			style = tab_style, selected_style = tab_selected_style,
			),
            dcc.Tab(label='LABEL INPUT', value='tab-2', children = 
			[
				html_comp.Div(children = 
				[
					dcc.Dropdown(id = 'label-dropdown', options = 
					[
						{'label': 'Blouse', 'value': 'Blouse'},
						{'label': 'Blazer', 'value': 'Blazer'},
						{'label': 'Button-Down', 'value': 'Button-Down'},
						{'label': 'Bomber', 'value': 'Bomber'},
						{'label': 'Anorak', 'value': 'Anorak'},
						{'label': 'Tee', 'value': 'Tee'},
						{'label': 'Tank', 'value': 'Tank'},
						{'label': 'Top', 'value': 'Top'},
						{'label': 'Sweater', 'value': 'Sweater'},
						{'label': 'Flannel', 'value': 'Flannel'},
						{'label': 'Hoodie', 'value': 'Hoodie'},
						{'label': 'Cardigan', 'value': 'Cardigan'},
						{'label': 'Jacket', 'value': 'Jacket'},
						{'label': 'Henley', 'value': 'Henley'},
						{'label': 'Poncho', 'value': 'Poncho'},
						{'label': 'Jersey', 'value': 'Jersey'},
						{'label': 'Turtleneck', 'value': 'Turtleneck'},
						{'label': 'Parka', 'value': 'Parka'},
						{'label': 'Peacoat', 'value': 'Peacoat'},
						{'label': 'Halter', 'value': 'Halter'},
						{'label': 'Skirt', 'value': 'Skirt'},
						{'label': 'Shorts', 'value': 'Shorts'},
						{'label': 'Jeans', 'value': 'Jeans'},
						{'label': 'Joggers', 'value': 'Joggers'},
						{'label': 'Sweatpants', 'value': 'Sweatpants'},
						{'label': 'Jeggings', 'value': 'Jeggings'},
						{'label': 'Cutoffs', 'value': 'Cutoffs'},
						{'label': 'Sweatshorts', 'value': 'Sweatshorts'},
						{'label': 'Leggings', 'value': 'Leggings'},
						{'label': 'Culottes', 'value': 'Culottes'},
						{'label': 'Chinos', 'value': 'Chinos'},
						{'label': 'Trunks', 'value': 'Trunks'},
						{'label': 'Sarong', 'value': 'Sarong'},
						{'label': 'Gauchos', 'value': 'Gauchos'},
						{'label': 'Jodhpurs', 'value': 'Jodhpurs'},
						{'label': 'Capris', 'value': 'Capris'},
						{'label': 'Dress', 'value': 'Dress'},
						{'label': 'Romper', 'value': 'Romper'},
						{'label': 'Coat', 'value': 'Coat'},
						{'label': 'Kimono', 'value': 'Kimono'},
						{'label': 'Jumpsuit', 'value': 'Jumpsuit'},
						{'label': 'Robe', 'value': 'Robe'},
						{'label': 'Caftan', 'value': 'Caftan'},
						{'label': 'Kaftan', 'value': 'Kaftan'},
						{'label': 'Coverup', 'value': 'Coverup'},
						{'label': 'Onesie', 'value': 'Onesie'}
					],
					persistence = True
					),
				],
				),
				html_comp.Div(id = 'output-image-upload-tab2')
			],
			style = tab_style, selected_style = tab_selected_style,
			),
        ], 
		),
            ])

def parse_contents_image(contents, recommendations):
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

def parse_contents_label(label, recommendations):
	return html_comp.Div(children = [
    html_comp.Div(children = [ 
        html_comp.H4(
                    children=html_comp.B('Selected category: {}'.format(label)),
                    style = {
                            'textAlign' : 'left'
                            },
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
	])

def recommendation(image):
    
	img = b64_to_pil(image[0].split("base64,")[-1])
	img = fastai.vision.Image(pil2tensor(img.convert('RGB'), np.float32).div_(255)).resize(224) 

	learner = load_learner(data, pretrained_model, model_metrics, model_path)
	saved_features = SaveFeatures(learner.model.module[1][4])

	_= learner.predict(img)

	embedding = saved_features.features

	similar_images_df = get_similar_images_annoy_centroid(annoy_tree, embedding[0], 30)

	output_data = show_similar_images(similar_images_df, learner, fig_size = (15, 15))
	return "data:image/png;base64,{}".format(output_data)

def dropdown(label):
	learner = load_learner(data, pretrained_model, model_metrics, model_path)
	similar_images_df = data_df[data_df['category'] == label].sample(30)
	output_data = show_similar_images(similar_images_df, learner, fig_size = (15, 15))
	return "data:image/png;base64,{}".format(output_data)

@app.callback([Output('output-image-upload-tab1', 'children'), Output('output-image-upload-tab2', 'children')],
              [Input('all-tabs-inline', 'value'), Input('upload-image', 'contents'), Input('label-dropdown', 'value')])

def update_output(tab, image, label):
	
	if tab == 'tab-1':
		if image is None:
			raise dash.exceptions.PreventUpdate()
		children = [parse_contents_image(image[0], recommendation(image))]
		return [children, {}]
	elif tab == 'tab-2':
		if label is None:
			raise dash.exceptions.PreventUpdate()
		children = [parse_contents_label(label, dropdown(label))]
		return [{}, children]

if __name__ == '__main__':
	app.run_server(debug = True)