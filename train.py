import model_build
import utils
import argparse

parser = argparse.ArgumentParser(description ='Train inputs')

parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'path of image folder')
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'available models: vgg16, densenet121')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate for training the model')
parser.add_argument('--epochs', type = int, default = 12, help = 'epochs for training the model')
parser.add_argument('--hidden_units', type = int, default = 2000, help = 'hidden units for training the model')
parser.add_argument('--gpu', type = str, default='cpu', help='gpu or cpu')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'checkpoints save folder')
    
args = parser.parse_args()

folder = args.data_dir
arch = args.arch
lr = args.learning_rate
epochs = args.epochs
hidden_units = args.hidden_units
device = args.gpu
save_folder = args.save_dir

train_data, trainloader, testloader, validloader = utils.load_data(folder)

model, criterion, optimizer = model_build.setup_nn(arch, hidden_units, lr)
model_build.train_nn(model, trainloader, validloader, criterion, optimizer, epochs, device)

model_build.save_checkpoint(save_folder, arch, hidden_units, lr, epochs)
     