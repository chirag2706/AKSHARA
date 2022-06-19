import numpy as np
from cv2 import cv2
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import copy
import random
from PIL import Image


def get_annotations(annotations):
    patterns = []
    pattern_dimensions = [0, 0]     # 0- width, 1- height
    antipatterns = []
    antipattern_dimensions = [0, 0]     # 0- width, 1- height

    for annotation in annotations:
        img = np.array(Image.open(annotation['document']).convert('L'))
        subregion = img[int(annotation['topY']):int(annotation['bottomY']), int(annotation['topX']):int(annotation['bottomX'])]
        if annotation['is_antipattern'] == True:
            antipatterns.append(subregion)
            antipattern_dimensions[0] += subregion.shape[1]
            antipattern_dimensions[1] += subregion.shape[0]
        
        else:
            patterns.append(subregion)
            pattern_dimensions[0] += subregion.shape[1]
            pattern_dimensions[1] += subregion.shape[0]
    
    pattern_dimensions[0] = int(pattern_dimensions[0]/len(patterns))
    pattern_dimensions[1] = int(pattern_dimensions[1]/len(patterns))

    if len(antipatterns) > 0:
        antipattern_dimensions[0] = int(antipattern_dimensions[0]/len(antipatterns))
        antipattern_dimensions[1] = int(antipattern_dimensions[1]/len(antipatterns))
    
    for i in range(len(patterns)):
        patterns[i] = cv2.resize(patterns[i], tuple(pattern_dimensions), interpolation=cv2.INTER_AREA)

    for i in range(len(antipatterns)):
        antipatterns[i] = cv2.resize(antipatterns[i], tuple(antipattern_dimensions), interpolation=cv2.INTER_AREA)

    patterns = np.array(patterns)
    antipatterns = np.array(antipatterns)

    return (patterns, pattern_dimensions, antipatterns, antipattern_dimensions)

def sp_noise(image, prob):
    # print(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i][j] = 255
            
    return image


"""
Algo by Chirag Gupta - CS18B006


Idea is to implement a new version of sp_noise function which will try to put white dots on black canvas in a smart manner

So, we will use Breadth First Search Algorithm and source point of this algo will be (149,149) which is the middle point of canvas (300 X 300) dimension

Reason of taking middle point of canvas is because letters will have higher probability on middle of canvas and we want to distort the image by randomly putting

white dots on 5% cells of a particular level. Their will be some kind of symmetry ,although it will not be exactly symmetric but it will place salt and pepper 

noise around letters which will ultimately lead to distortion of letter

"""

# def sp_noise(image,threshold):
#     # print("Chirag's sp_noise function is called")
#     def isValid(currX,currY,rows,cols):
#         return (currX>=0 and currY>=0 and currX<rows and currY<cols)

#     newImage = np.array(image)
#     # print(newImage.shape)

#     rows = newImage.shape[0]
#     cols = newImage.shape[1]

#     vis = []

#     for i in range(rows):
#         temp = []
#         for j in range(cols):
#             temp.append(False)
#         vis.append(temp)

#     srcX = rows//2
#     srcY = cols//2

#     #idea is to use BFS algo

#     queueOfVisitedCells = []

#     queueOfVisitedCells.append([srcX,srcY])
#     vis[srcX][srcY] = True
#     prevLevel = 1
#     currLevel = 0

#     newImage[srcX][srcY] = 255 # middle element on canvas will always be white cell

#     arrX = [0,-1,-1,-1,0,1,1,1]
#     arrY = [-1,-1,0,1,1,1,0,-1]

#     totalDirections = len(arrX)

#     while(len(queueOfVisitedCells)>0):
#         currLevel = 0
#         while(prevLevel>0):
#             prevLevel-=1

      
#             currCell = queueOfVisitedCells[0]
#             queueOfVisitedCells.pop(0)
#             currX = currCell[0]
#             currY = currCell[1]
#             #we will try to visit all unvisited neighbours of current cell
      
#             for directionId in range(totalDirections):
#                 newCurrX = currX + arrX[directionId]
#                 newCurrY = currY + arrY[directionId]

#                 if(isValid(newCurrX,newCurrY,rows,cols) and (not vis[newCurrX][newCurrY])):
#                     vis[newCurrX][newCurrY] = True
#                     queueOfVisitedCells.append([newCurrX,newCurrY])
#                     currLevel+=1

#         prevLevel = currLevel
    
#         #now, all cells which are present in queue are the once from which we need to randomly assign white dots
#         for idx in range(currLevel):
#             rdn = random.random()
#             if(rdn<threshold):
#                 newImage[queueOfVisitedCells[idx][0]][queueOfVisitedCells[idx][1]] = 255
  
#     return newImage

def perturb(x1=None,level=1):
    x = copy.deepcopy(x1)
    x1 = np.array(x1)
    #----------------#
    def rand_neighbor(i,j,r,c):
    
        i1 = int(i + level*np.random.randn())
        j1 = int(j + level*np.random.randn())

        i1 = i if (i1<0 or i1>=r) else i1
        j1 = j if (j1<0 or j1>=c) else j1

        return i1,j1

    #----------------#
    r = x1.shape[0]
    c = x1.shape[1]

    x2 = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            i2,j2 = rand_neighbor(i,j,r,c)
            x2[i2][j2] = x[i][j]
    return x2



def generate_pattern_canvas(canvas_height, canvas_width, p_dataset, p_dimension, probability):
    canvas = np.zeros((canvas_height, canvas_width))
    canvas = sp_noise(canvas, 0.05)
    labels = np.zeros((canvas_height, canvas_width))
    
    i = 5
    while i < canvas_height:
        j = 5
        while j < canvas_width:
            prob = random.uniform(0,1)
            if(prob <= probability):
                number_of_images = len(p_dataset)
                random_image_number = random.randint(0, number_of_images - 1)
                sub_image = p_dataset[random_image_number]

                (_, sub_image) = cv2.threshold(sub_image, 127, 255, cv2.THRESH_BINARY)
                sub_image = perturb(sub_image,0.5)
                sub_image = sp_noise(sub_image, 0.05)
                # print(sub_image.shape)
                #sub_image[np.where(np.all(sub_image[..., :3] == 255, -1))] = 0 #line which make background of image transparent

                if(i+p_dimension[1] < canvas_height and j+p_dimension[0] < canvas_width):
                    canvas[i:i+p_dimension[1], j:j+p_dimension[0]] = sub_image
                    labels[i+(p_dimension[1]//2), j+(p_dimension[0]//2)] = 199920

            j += p_dimension[1] + 5
        i += p_dimension[0] + 5
        
    canvas[canvas <= 127] = 0
    canvas[canvas > 127] = 255
    # prob = random.uniform(0,1)
    # if(prob <= 0.5):
    # canvas = perturb(canvas,0.2) # to distort canvas
    return (canvas, labels)

def generate_antipattern_canvas(canvas_height, canvas_width, ap_dataset, ap_dimensions, p_dimension, anc, apc):
    canvas = np.zeros((canvas_height, canvas_width))
    canvas = sp_noise(canvas, 0.05)
    labels = np.zeros((canvas_height, canvas_width))

    i = 5
    while i < canvas_height:
        max_height = 0
        j = 5
        while j < canvas_width:
            temp_height = 0
            temp_width = 0
            if anc == -1 or len(ap_dataset) == 0:
                sub_image = np.zeros((p_dimension[1], p_dimension[0]))
                temp_height = p_dimension[1]
                temp_width = p_dimension[0]
            elif len(ap_dataset) > 0:
                sub_image = ap_dataset[anc]
                temp_height = ap_dimensions[1]
                temp_width = ap_dimensions[0]

            (_, sub_image) = cv2.threshold(sub_image, 127, 255, cv2.THRESH_BINARY)
            #sub_image[np.where(np.all(sub_image[..., :3] == 255, -1))] = 0 #line which make background of image transparent
            if(i+temp_height < canvas_height and j+temp_width < canvas_width):
                canvas[i:i+temp_height, j:j+temp_width] = sub_image

            if temp_height > max_height:
                max_height = temp_height

            j += temp_width + 5
            anc += 1
            if anc == len(ap_dataset):
                anc = -1
                apc += 1
        
        i += max_height + 5
    #canvas = perturb(canvas,0.5) # to distort canvas
    return canvas, labels, anc, apc

def make_dataset(p_dataset, p_dimension, ap_dataset, ap_dimensions, no_of_images = 150, probability = 0.6):
    X = []
    y = []

    anc = -1
    apc = 0
    for i in range(no_of_images):
        if apc < 60:
            image, label, anc, apc = generate_antipattern_canvas(300, 300, ap_dataset, ap_dimensions, p_dimension, anc, apc)
        else:
            image,label =  generate_pattern_canvas(300, 300, p_dataset, p_dimension, probability)
        X.append([image])
        y.append(label.reshape(1, -1))

    X, y = torch.from_numpy(np.stack(X)), torch.from_numpy(np.vstack(y))
    return (X, y)

class cnn(nn.Module):
    def __init__(self, kernel_dimension):
        super(cnn,self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 10, (kernel_dimension[0], kernel_dimension[1]), padding = ((kernel_dimension[0]-1)//2, (kernel_dimension[1]-1)//2)),
            nn.ReLU(),
            nn.Conv2d(10, 1, 1, padding=(0,0)),
            nn.ReLU(),
        )

    def forward(self,x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        return x

def train_model(train_loader, kernel_dimension = [30, 30], max_epochs = 1):
    first_epoch_loss = None
    last_epoch_loss  = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = cnn(kernel_dimension)

    net.to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(net.parameters(), lr = 0.001)
    best_model = None
    min_loss = sys.maxsize

    for epoch in range(max_epochs):
        for i, data in enumerate(train_loader, 0):
            X, y = data
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = net(X)
            loss = loss_fn(out, y)

            if(loss < min_loss):
                best_model = copy.deepcopy(net)
                min_loss = loss

            loss.backward()
            opt.step()

        print('Epoch: {: >2}/{: >2}  loss: {}'.format(epoch, max_epochs, loss))

        if(epoch == 0):
            first_epoch_loss = int(loss)
        elif(epoch == max_epochs - 1):
            last_epoch_loss = int(loss)
    
    return (best_model, min_loss, first_epoch_loss, last_epoch_loss)

def lookup(annotations):
    patterns, pattern_dimensions, antipatterns, antipattern_dimensions = get_annotations(annotations)
    patterns = 255 - patterns
    antipatterns = 255 - antipatterns

    (train_x, train_y) = make_dataset(p_dataset = patterns, p_dimension = pattern_dimensions, ap_dataset = antipatterns, ap_dimensions = antipattern_dimensions, no_of_images = 200)
    train_x = train_x.type(torch.float32)
    train_y = train_y.type(torch.float32)
    train_dataset = data.TensorDataset(train_x, train_y)
    train_loader = data.DataLoader(train_dataset, batch_size = 4, shuffle = True)
    
    kernel_width = pattern_dimensions[0]
    kernel_height = pattern_dimensions[1]

    if(kernel_height%2 == 0):
        kernel_height += 1

    if(kernel_width%2 == 0) :
        kernel_width += 1

    totalIterations = 2
    currentIteration = 0
    true_learning = False
    while currentIteration < totalIterations:
        print('Iteration =>', currentIteration)
        currentIteration += 1
        best_model, loss, first_loss, last_loss = train_model(train_loader, max_epochs = 20, kernel_dimension = [kernel_height, kernel_width])

        """
            Chirag Gupta - CS18B006 has commented this code
        """
        if((first_loss*0.3) >= last_loss):
            true_learning = True
            break
    true_learning = True
    print('Final Loss =',int(loss))

    if(true_learning):
        torch.save(best_model.state_dict(), 'trained_models/model.pth')
        return (True, [kernel_height, kernel_width])
    else:
        return (False, [0,0])