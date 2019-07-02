import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from PIL import Image

def setup_nn(arch = 'vgg16', hidden_units = 2000, lr = 0.001):
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_layer = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_layer = 1024
    else:
        print("{} is not a valid model. (Available models: vgg16, densenet121)".format(arch))
        
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_layer, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 1000)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(1000, 500)),
                              ('relu3', nn.ReLU()),
                              ('fc4', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    return model, criterion, optimizer



def validation(model, validloader, criterion, optimizer):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validloader:

        inputs, labels = inputs.to('cuda:0') , labels.to('cuda:0')
        model.to('cuda:0')
        
        optimizer.zero_grad()

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy



def train_nn(model, trainloader, validloader, criterion, optimizer, epochs = 12, device='cpu'):
    
    epochs = epochs
    print_every = 20
    steps = 0

    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda:0')

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, optimizer)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                model.train() 
                
def save_checkpoint(train_data, save_folder='checkpoint.pth', arch='vgg16', hidden_units=2000, lr=0.001, epochs=12):
    
    if arch == 'vgg16':
        input_size = 25088
    elif arch == 'densenet121':
        input_size = 1024
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'structure' : arch,
                  'input_size': input_size,
                  'hidden_layer1': hidden_units,
                  'lr': lr,
                  'epochs': epochs,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, save_folder)
    
def load_checkpiont(path):
    checkpoint = torch.load(path)
    arch = checkpoint['structure']
    hidden_units = checkpoint['hidden_layer1']
    lr = checkpoint['lr']
    model = setup_nn(arch, hidden_units, lr)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(img):
    image = Image.open(img)

    image = image.resize((256, 256), Image.ANTIALIAS)
    image = image.crop((16, 16, 240, 240))
    
    np_image = np.array(image)
    np_image = np_image/225
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    r_np_image = (np_image[:,:,0] - mean[0]) / std[0]
    g_np_image = (np_image[:,:,1] - mean[1]) / std[1]
    b_np_image = (np_image[:,:,2] - mean[2]) / std[2]
    
    np_image[:,:,0] = r_np_image
    np_image[:,:,1] = g_np_image
    np_image[:,:,2] = b_np_image
    
    np_image = np.transpose(np_image, (2,0,1))
    
    return np_image

def predict(image_path, model, top_k = 3, device = 'cpu'):
    
    if torch.cuda.is_available() and device == 'gpu':
        model.to('cuda:0')
        
    classes_map = {v: k for k, v in model.class_to_idx.items()}
    
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    
    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img)
        
    prob, idx = torch.topk(output, topk)
    prob = prob.exp()
    prob = prob.numpy()[0]
    idx = idx.numpy()[0]
    
    classes = [classes_map[x] for x in idx]
    
    return prob, classes

