import torch
from torch.utils.data import Dataset
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as tfms
from PIL import Image
from skimage.draw import random_shapes


class CustomDataset(Dataset):
    def __init__(self, image_paths, image_labels=None):
        
        self.images = [Image.open(path).convert("L") for path in image_paths]
        self.labels = image_labels

    def __len__(self):
       
        return len(self.images)

    ### converts image to suitable format for pytorch
    def transform_image(self, image):
        transform = tfms.Compose([
            tfms.ToTensor(),
            tfms.Resize((256,256),antialias=True)
        ])
        return transform(image)

    ### adds the pesky random shapes
    def add_noise(self,img):
        noise, _ = random_shapes(img.shape[-2:],channel_axis=None, min_shapes=10,max_shapes=20,
                                 min_size=20,max_size=30,allow_overlap=False,intensity_range=(0, 100))
        
        noise = torch.from_numpy(noise).float()/255.0

        img = torch.where(noise < 0.9, img + (1-noise) , img)

        return img

    def __getitem__(self, idx):

        input_img = self.images[idx]
        input_img = self.transform_image(input_img)
    
        targ_img = input_img.clone() # We want to predict the original image

        input_img = self.add_noise(input_img)
        
        return input_img.float(), targ_img.float()