import model_build
import utils
import argparse

parser = argparse.ArgumentParser(description ='Predict inputs')

parser.add_argument('--input', type = str, help = 'path of input image')
parser.add_argument('--checkpoint', type = str, help = 'path of checkpoint file')
parser.add_argument('--top_k', type = int, default = 3, help = 'number of top probabilities to show')
parser.add_argument('--category_names', type = str, help = 'path of category names file')
parser.add_argument('--gpu', type = str, default = 'cpu', help = 'gpu or cpu')
    
args = parser.parse_args()

img = args.input
checkpoint = args.checkpoint
top_k = args.top_k
cat_to_names = args.category_names
device = args.gpu

with open(args.category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)
    
    
model = model_build.load_checkpiont(checkpoint)
probs, classes = model_build.predict(img, model, top_k, device)

a = [cat_to_name[str(x + 1)] for x in np.array(classes)]
b = np.array(probs)

for i in range(top_k):
	print("{} - {}".format(a[i], b[i]))

     