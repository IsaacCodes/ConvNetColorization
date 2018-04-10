<!DOCTYPE html>
<html>
<body>
<pre>&quot;&quot;&quot;
Colourization of CIFAR-10 Horses via classification.
&quot;&quot;&quot;

from __future__ import print_function
import argparse
import os
import math
import numpy as np
import numpy.random as npr
import scipy.misc
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
matplotlib.use(&#39;Agg&#39;) # switch backend
import matplotlib.pyplot as plt 


from load_data import load_cifar10

HORSE_CATEGORY = 7

######################################################################
# Data related code
######################################################################
def get_rgb_cat(xs, colours):
    &quot;&quot;&quot;
    Get colour categories given RGB values. This function doesn&#39;t
    actually do the work, instead it splits the work into smaller
    chunks that can fit into memory, and calls helper function
    _get_rgb_cat

    Args:
      xs: float numpy array of RGB images in [B, C, H, W] format
      colours: numpy array of colour categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    &quot;&quot;&quot;
    if np.shape(xs)[0] &lt; 100:
        return _get_rgb_cat(xs)
    batch_size = 100
    nexts = []
    for i in range(0, np.shape(xs)[0], batch_size):
        next = _get_rgb_cat(xs[i:i+batch_size,:,:,:], colours)
        nexts.append(next)
    result = np.concatenate(nexts, axis=0)
    return result

def _get_rgb_cat(xs, colours):
    &quot;&quot;&quot;
    Get colour categories given RGB values. This is done by choosing
    the colour in `colours` that is the closest (in RGB space) to
    each point in the image `xs`. This function is a little memory
    intensive, and so the size of `xs` should not be too large.

    Args:
      xs: float numpy array of RGB images in [B, C, H, W] format
      colours: numpy array of colour categories and their RGB values
    Returns:
      result: int numpy array of shape [B, 1, H, W]
    &quot;&quot;&quot;
    num_colours = np.shape(colours)[0]
    xs = np.expand_dims(xs, 0)
    cs = np.reshape(colours, [num_colours,1,3,1,1])
    dists = np.linalg.norm(xs-cs, axis=2) # 2 = colour axis
    cat = np.argmin(dists, axis=0)
    cat = np.expand_dims(cat, axis=1)
    return cat

def get_cat_rgb(cats, colours):
    &quot;&quot;&quot;
    Get RGB colours given the colour categories

    Args:
      cats: integer numpy array of colour categories
      colours: numpy array of colour categories and their RGB values
    Returns:
      numpy tensor of RGB colours
    &quot;&quot;&quot;
    return colours[cats]

def process(xs, ys, max_pixel=256.0):
    &quot;&quot;&quot;
    Pre-process CIFAR10 images by taking only the horse category,
    shuffling, and have colour values be bound between 0 and 1

    Args:
      xs: the colour RGB pixel values
      ys: the category labels
      max_pixel: maximum pixel value in the original data
    Returns:
      xs: value normalized and shuffled colour images
      grey: greyscale images, also normalized so values are between 0 and 1
    &quot;&quot;&quot;
    xs = xs / max_pixel
    xs = xs[np.where(ys == HORSE_CATEGORY)[0], :, :, :]
    npr.shuffle(xs)
    grey = np.mean(xs, axis=1, keepdims=True)
    return (xs, grey)

def get_batch(x, y, batch_size):
    &#39;&#39;&#39;
    Generated that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    &#39;&#39;&#39;
    N = np.shape(x)[0]
    assert N == np.shape(y)[0]
    for i in range(0, N, batch_size):
        batch_x = x[i:i+batch_size, :,:,:]
        batch_y = y[i:i+batch_size, :,:,:]
        yield (batch_x, batch_y)

def plot(input, gtlabel, output, colours, path):
    &quot;&quot;&quot;
    Generate png plots of input, ground truth, and outputs

    Args:
      input: the greyscale input to the colourization CNN
      gtlabel: the grouth truth categories for each pixel
      output: the predicted categories for each pixel
      colours: numpy array of colour categories and their RGB values
      path: output path
    &quot;&quot;&quot;
    grey = np.transpose(input[:10,:,:,:], [0,2,3,1])
    gtcolor = get_cat_rgb(gtlabel[:10,0,:,:], colours)
    predcolor = get_cat_rgb(output[:10,0,:,:], colours)

    img = np.vstack([
      np.hstack(np.tile(grey, [1,1,1,3])),
      np.hstack(gtcolor),
      np.hstack(predcolor)])
    scipy.misc.toimage(img, cmin=0, cmax=1).save(path)


######################################################################
# MODELS
######################################################################

class MyConv2d(nn.Module):
    &quot;&quot;&quot;
    Our simplified implemented of nn.Conv2d module for 2D convolution
    &quot;&quot;&quot;
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding is None:
            self.padding = kernel_size // 2
        else:
            self.padding = padding
        self.weight = nn.parameter.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, padding=self.padding)

class MyDilatedConv2d(MyConv2d):
    &quot;&quot;&quot;
    Dilated Convolution 2D
    &quot;&quot;&quot;
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(MyDilatedConv2d, self).__init__(in_channels,
                                              out_channels,
                                              kernel_size)
        self.dilation = dilation

    def forward(self, input):
        ############### YOUR CODE GOES HERE ############### 
        pass
        ###################################################

class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colours):
        super(CNN, self).__init__()
        padding = kernel // 2
        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU()
        )
        self.rfconv = nn.Sequential(
            MyConv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU()
        )
        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters * 2, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            MyConv2d(num_filters, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )
        self.finalconv = MyConv2d(num_colours, num_colours, kernel_size=kernel)

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        self.out_final = self.finalconv(self.out5)
        return self.out_final


###################UNet implementations#########################
class UNet(nn.Module):
    def __init__(self, kernel, num_filters, num_colours):
        super(UNet, self).__init__()

        padding = kernel // 2
        self.downconv1 = nn.Sequential(
            MyConv2d(1, num_filters, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.downconv2 = nn.Sequential(
            MyConv2d(num_filters, num_filters*2, kernel_size=kernel, padding=padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU()
        )
        self.rfconv = nn.Sequential(
            MyConv2d(num_filters*2, num_filters*2, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU()
        )
        self.upconv1 = nn.Sequential(
            MyConv2d(num_filters * 4, num_filters, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            MyConv2d(num_filters*2, num_colours, kernel_size=kernel, padding=padding),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(num_colours),
            nn.ReLU()
        )
        self.finalconv = MyConv2d(num_colours+1, num_colours, kernel_size=kernel)


    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(torch.cat((self.out3,self.out2),1))
        self.out5 = self.upconv2(torch.cat((self.out4, self.out1),1))
        self.out_final = self.finalconv(torch.cat((self.out5,x),1))
        return self.out_final


class DilatedUNet(UNet):
    def __init__(self, kernel, num_filters, num_colours):
        super(DilatedUNet, self).__init__(kernel, num_filters, num_colours)
        # replace the intermediate dilations
        self.rfconv = nn.Sequential(
            MyDilatedConv2d(num_filters*2, num_filters*2, kernel_size=kernel, dilation=1),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU())

######################################################################
# Torch Helper
######################################################################

def get_torch_vars(xs, ys, gpu=False):
    &quot;&quot;&quot;
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tenosor): greyscale input
      ys (int numpy tenosor): categorical labels 
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      Variable(xs), Variable(ys)
    &quot;&quot;&quot;
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).long()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return Variable(xs), Variable(ys)

def compute_loss(criterion, outputs, labels, batch_size, num_colours):
    &quot;&quot;&quot;
    Helper function to compute the loss. Since this is a pixelwise
    prediction task we need to reshape the output and ground truth
    tensors into a 2D tensor before passing it in to the loss criteron.

    Args:
      criterion: pytorch loss criterion
      outputs (pytorch tensor): predicted labels from the model
      labels (pytorch tensor): ground truth labels
      batch_size (int): batch size used for training
      num_colours (int): number of colour categories
    Returns:
      pytorch tensor for loss
    &quot;&quot;&quot;

    loss_out = outputs.transpose(1,3) \
                      .contiguous() \
                      .view([batch_size*32*32, num_colours])
    loss_lab = labels.transpose(1,3) \
                      .contiguous() \
                      .view([batch_size*32*32])
    return criterion(loss_out, loss_lab)

def run_validation_step(cnn, criterion, test_grey, test_rgb_cat, batch_size,
                        colour, plotpath=None):
    correct = 0.0
    total = 0.0
    losses = []
    for i, (xs, ys) in enumerate(get_batch(test_grey,
                                           test_rgb_cat,
                                           batch_size)):
        images, labels = get_torch_vars(xs, ys, args.gpu)
        outputs = cnn(images)

        val_loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
        losses.append(val_loss.data[0])

        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        total += labels.size(0) * 32 * 32
        correct += (predicted == labels.data).sum()

    if plotpath: # only plot if a path is provided
        plot(xs, ys, predicted.cpu().numpy(), colours, plotpath)

    val_loss = np.mean(losses)
    val_acc = 100 * correct / total
    return val_loss, val_acc


######################################################################
# MAIN
######################################################################

if __name__ == &#39;__main__&#39;:
    parser = argparse.ArgumentParser(description=&quot;Train colourization&quot;)
    parser.add_argument(&#39;--gpu&#39;, action=&#39;store_true&#39;, default=False,
                        help=&quot;Use GPU for training&quot;)
    parser.add_argument(&#39;--valid&#39;, action=&quot;store_true&quot;, default=False,
                        help=&quot;Perform validation only (don&#39;t train)&quot;)
    parser.add_argument(&#39;--checkpoint&#39;, default=&quot;&quot;,
                        help=&quot;Model file to load and save&quot;)
    parser.add_argument(&#39;--plot&#39;, action=&quot;store_true&quot;, default=False,
                        help=&quot;Plot outputs every epoch during training&quot;)
    parser.add_argument(&#39;-c&#39;, &#39;--colours&#39;,
                        default=&#39;colours/colour_kmeans24_cat7.npy&#39;,
                        help=&quot;Discrete colour clusters to use&quot;)
    parser.add_argument(&#39;-m&#39;, &#39;--model&#39;, choices=[&quot;CNN&quot;, &quot;UNet&quot;, &quot;DUNet&quot;],
                        help=&quot;Model to run&quot;)
    parser.add_argument(&#39;-k&#39;, &#39;--kernel&#39;, default=3, type=int,
                        help=&quot;Convolution kernel size&quot;)
    parser.add_argument(&#39;-f&#39;, &#39;--num_filters&#39;, default=32, type=int,
                        help=&quot;Base number of convolution filters&quot;)
    parser.add_argument(&#39;-l&#39;, &#39;--learn_rate&#39;, default=0.001, type=float,
                        help=&quot;Learning rate&quot;)
    parser.add_argument(&#39;-b&#39;, &#39;--batch_size&#39;, default=100, type=int,
                        help=&quot;Batch size&quot;)
    parser.add_argument(&#39;-e&#39;, &#39;--epochs&#39;, default=25, type=int,
                        help=&quot;Number of epochs to train&quot;)
    parser.add_argument(&#39;-s&#39;, &#39;--seed&#39;, default=0, type=int,
                        help=&quot;Numpy random seed&quot;)

    args = parser.parse_args()

    # Set the maximum number of threads to prevent crash in Teaching Labs
    torch.set_num_threads(5)

    # Numpy random seed
    npr.seed(args.seed)

    # LOAD THE COLOURS CATEGORIES
    colours = np.load(args.colours)[0]
    num_colours = np.shape(colours)[0]

    # LOAD THE MODEL
    if args.model == &quot;CNN&quot;:
        cnn = CNN(args.kernel, args.num_filters, num_colours)
    elif args.model == &quot;UNet&quot;:
        cnn = UNet(args.kernel, args.num_filters, num_colours)
    else: # model == &quot;DUNet&quot;:
        cnn = DilatedUNet(args.kernel, args.num_filters, num_colours)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)

    # DATA
    print(&quot;Loading data...&quot;)
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print(&quot;Transforming data...&quot;)
    train_rgb, train_grey = process(x_train, y_train)
    train_rgb_cat = get_rgb_cat(train_rgb, colours)
    test_rgb, test_grey = process(x_test, y_test)
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # Create the outputs folder if not created already
    if not os.path.exists(&quot;outputs&quot;):
        os.makedirs(&quot;outputs&quot;)

    # Run validation only
    if args.valid:
        if not args.checkpoint:
            raise ValueError(&quot;You need to give trained model to evaluate&quot;)

        print(&quot;Loading checkpoint...&quot;)
        cnn.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
        img_path = &quot;outputs/eval_%s.png&quot; % args.model
        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                img_path)
        print(&#39;Evaluating Model %s: %s&#39; % (args.model, args.checkpoint))
        print(&#39;Val Loss: %.4f, Val Acc: %.1f%%&#39; % (val_loss, val_acc))
        print(&#39;Sample output available at: %s&#39; % img_path)
        exit(0)

    print(&quot;Beginning training ...&quot;)
    if args.gpu: cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        cnn.train() # Change model to &#39;train&#39; mode
        losses = []
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = compute_loss(criterion,
                                outputs,
                                labels,
                                batch_size=args.batch_size,
                                num_colours=num_colours)
            loss.backward()
            optimizer.step()
            losses.append(loss.data[0])

        # plot training images
        if args.plot:
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(xs, ys, predicted.cpu().numpy(), colours,
                 &#39;outputs/train_%d.png&#39; % epoch)

        # plot training images
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print(&#39;Epoch [%d/%d], Loss: %.4f, Time (s): %d&#39; % (
            epoch+1, args.epochs, avg_loss, time_elapsed))

        # Evaluate the model
        cnn.eval()  # Change model to &#39;eval&#39; mode (BN uses moving mean/var).

        outfile = None
        if args.plot:
            outfile = &#39;outputs/test_%d.png&#39; % epoch

        val_loss, val_acc = run_validation_step(cnn,
                                                criterion,
                                                test_grey,
                                                test_rgb_cat,
                                                args.batch_size,
                                                colours,
                                                outfile)

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print(&#39;Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d&#39; % (
            epoch+1, args.epochs, val_loss, val_acc, time_elapsed))

    # Plot training curve
    plt.plot(train_losses, &quot;ro-&quot;, label=&quot;Train&quot;)
    plt.plot(valid_losses, &quot;go-&quot;, label=&quot;Validation&quot;)
    plt.legend()
    plt.title(&quot;Loss&quot;)
    plt.xlabel(&quot;Epochs&quot;)
    plt.savefig(&quot;outputs/training_curve.png&quot;)

    if args.checkpoint:
        print(&#39;Saving model...&#39;)
        torch.save(cnn.state_dict(), args.checkpoint)
</pre>
</body>
</html>
