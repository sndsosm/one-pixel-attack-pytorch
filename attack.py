import os
import sys
import numpy as np

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

args = parser.parse_args()
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict

from utee import misc
print = misc.logger.info

model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def cifar10(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model
def perturb_image(xs, img):
  if xs.ndim < 2:
    xs = np.array([xs])
  batch = len(xs)
  imgs = img.repeat(batch, 1, 1, 1)
  xs = xs.astype(int)
  count = 0
  for x in xs:
    pixels = np.split(x, len(x)/9)
    for pixel in pixels:
      x_pos1, y_pos1, red, x_pos2, y_pos2, green, x_pos3, y_pos3, blue= pixel
      imgs[count, 0, x_pos1, y_pos1] = (red/255.0-0.4914)/0.2023
      imgs[count, 1, x_pos2, y_pos2] = (green/255.0-0.4822)/0.1994
      imgs[count, 2, x_pos3, y_pos3] = (blue/255.0-0.4465)/0.2010
    count += 1
  return imgs

def predict_classes(xs, img, target_calss, net, minimize=True):
  imgs_perturbed = perturb_image(xs, img.clone())
  input = Variable(imgs_perturbed, volatile=True).cuda()
  predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_calss]

  return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):

  attack_image = perturb_image(x, img.clone())
  input = Variable(attack_image, volatile=True).cuda()
  confidence = F.softmax(net(input)).data.cpu().numpy()[0]
  predicted_class = np.argmax(confidence)

  if (verbose):
    print ("Confidence: %.4f"%confidence[target_calss])
  if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
    return True

def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):

  targeted_attack = target is not None
  target_calss = target if targeted_attack else label
  bounds = [(0,32), (0,32),(0,255), (0,32), (0,32),(0,255),(0,32), (0,32),(0,255)] * pixels
  popmul = max(1, popsize//len(bounds))
  predict_fn = lambda xs: predict_classes(xs, img, target_calss, net, target is None)
  callback_fn = lambda x, convergence: attack_success(x, img, target_calss, net, targeted_attack, verbose)
  inits = np.zeros([popmul*len(bounds), len(bounds)])
  for init in inits:
    for i in range(pixels):
      init[i*5+0] = np.random.random()*32
      init[i*5+1] = np.random.random()*32
      init[i*5+2] = np.random.normal(128,127)
      init[i*5+3] = np.random.random()*32
      init[i*5+4] = np.random.random()*32
      init[i*5+5] = np.random.normal(128,127)
      init[i*5+6] = np.random.random()*32
      init[i*5+7] = np.random.random()*32
      init[i*5+8] = np.random.normal(128,127)
  attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
    recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)
  attack_image = perturb_image(attack_result.x, img)
  attack_var = Variable(attack_image, volatile=True).cuda()
  predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]
  predicted_class = np.argmax(predicted_probs)
  if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
    return 1, attack_result.x.astype(int)
  return 0, [None]

def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

  correct = 0
  success = 0

  for batch_idx, (input, target) in enumerate(loader):

    img_var = Variable(input, volatile=True).cuda()
    prior_probs = F.softmax(net(img_var))
    _, indices = torch.max(prior_probs, 1)
    
    if target[0] != indices.data.cpu()[0]:
      continue

    correct += 1
    target = target.numpy()

    targets = [None] if not targeted else range(10)

    for target_calss in targets:
      if (targeted):
        if (target_calss == target[0]):
          continue
      
      flag, x = attack(input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)

      success += flag
      if (targeted):
        success_rate = float(success)/(9*correct)
      else:
        success_rate = float(success)/correct

      if flag == 1:
        print ("success rate: %.4f (%d/%d) [(x1,y1) = (%d,%d),(x2,y2) = (%d,%d),(x3,y3) = (%d,%d) and (R,G,B)=(%d,%d,%d)]"%(
          success_rate, success, correct, x[0],x[1],x[3],x[4],x[6],x[7],x[2],x[5],x[8]))
    
    if correct == args.samples:
      break

  return success_rate

def main():

  print ("==> Loading data and model...")
  tranfrom_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranfrom_test)
  testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

  class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  net = cifar10(128, pretrained='log/cifar10/best-135.pth')
  net.cuda()
  cudnn.benchmark = True

  print ("==> Starting attck...")

  results = attack_all(net, testloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
  print ("Final success rate: %.4f"%results)

if __name__ == '__main__':
  main()
