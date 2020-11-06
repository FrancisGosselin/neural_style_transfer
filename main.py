
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

content_filename = "content/wallpaper.jpg"
style_filename = "styles/autumn.jpg"
result_filename = "results/wallpaper_autumn.jpg"

w = 224
h = 224
vgg19_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
vgg19_std = torch.tensor([0.229, 0.224, 0.225], device=device)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(w),
    transforms.ToTensor(),
    transforms.Normalize(mean=vgg19_mean, std=vgg19_std),
])

preprocess_without_normalize = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(w),
    transforms.ToTensor(),
])


vgg19 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
vgg19.to(device)
vgg19.train()
# model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19_bn', pretrained=True)

# Make hooks
style_features_layers = [vgg19.features[0], vgg19.features[5], vgg19.features[10], vgg19.features[19], vgg19.features[28]]
content_features_layers = [vgg19.features[21]]

def get_activation(output_dict, output_index):
    def hook(model, input, output):
        output_dict[output_index] = output
    return hook

style_features_output = {}
content_features_output = {}

for i, layer in enumerate(style_features_layers):
    layer.register_forward_hook(get_activation(style_features_output, i))

for i, layer in enumerate(content_features_layers):
    layer.register_forward_hook(get_activation(content_features_output, i))


def get_white_noise_image(w, h):
    return np.array([ np.full((3,), random.random() , dtype='double') for i in range(w*h)]).reshape((w,h,3)).transpose((2, 0, 1))

def content_loss(base_content, target):
    return torch.mean(torch.square(base_content - target))


def gram_matrix(base_input):
    batch_size, channels, H, W = base_input.size()
    a = base_input.view(channels, -1)
    n = H*W*channels
    gram = torch.matmul( a, torch.t(a))
    return gram / n


def style_loss(base_style, target):
    return torch.mean(torch.square(gram_matrix(base_style) - target))

def total_variation_loss(x):
    a = torch.square(x[:, :h-1, :w-1] - x[:, 1:, :w-1])
    b = torch.square(x[:, :h-1, :w-1] - x[:, :h-1, 1:])
    return torch.sum(a + b)

def compute_loss(base_image, style_features, content_features, content_weight=1, style_weight = 200000, total_variation_weight = 0.5):
    vgg19(base_image.unsqueeze(0))
    content_sum = torch.tensor(0.0, device=device)
    for i, content_feature in content_features.items():
        content_sum += content_loss(content_features_output[i], content_feature)

    style_sum = torch.tensor(0.0, device=device)
    for i, style_feature in style_features.items():
        style_sum += style_loss(style_features_output[i], style_feature)

    return content_weight*content_sum + style_weight*style_sum #+ total_variation_weight*total_variation_loss(base_image)
    

def get_style_content_features(content_image, style_image):
    content_image_features = {}
    vgg19(content_image.unsqueeze(0))
    for i, content_feature in content_features_output.items():
         content_image_features[i] = content_feature.clone().detach()

    style_image_features = {}
    vgg19(style_image.unsqueeze(0))
    for i, style_feature in style_features_output.items():
         style_image_features[i] = gram_matrix(style_feature.clone().detach())
        
    return content_image_features, style_image_features


content_tensor = preprocess(Image.open(content_filename)).to(device)
style_tensor = preprocess(Image.open(style_filename)).to(device)
base_tensor = preprocess_without_normalize(Image.open(content_filename)).to(device)
base_tensor.requires_grad = True

content_image_features, style_image_features = get_style_content_features(content_tensor, style_tensor)
optimizer = torch.optim.Adam([base_tensor], lr=1e-2)
iterations = 1000

print(f"Using device {device}")

for i in progressbar(range(iterations), "Iteration: ", 40):
    optimizer.zero_grad()
    loss = compute_loss((base_tensor - vgg19_mean.view(-1,1,1))/vgg19_std.view(-1,1,1), style_image_features, content_image_features)
    loss.backward()
    optimizer.step()

    base_tensor.data.clamp_(0,1)

plt.imshow(base_tensor.permute(1, 2, 0).cpu().detach().numpy())
plt.savefig(result_filename)
