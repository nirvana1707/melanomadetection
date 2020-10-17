#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import imageio
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[3]:


import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from albumentations.pytorch.transforms import ToTensorV2


# In[5]:


from sklearn import model_selection
from sklearn import metrics


# In[6]:


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionv4(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['inceptionv4'][pretrained]
        assert num_classes == settings['num_classes'],             "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        #model.load_state_dict(model_zoo.load_url(settings['url']))
        path_weight = "/kaggle/input/pytorch-pretrained-models/inceptionv4-8e4777a0.pth"
        model.load_state_dict(torch.load(path_weight))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV4(num_classes=num_classes)
    return model


# In[7]:


class stratification:
    def __init__(self,path_dir,num_splits):
        self.input_path = path_dir
        self.n_splits = num_splits
        self.df = pd.read_csv(os.path.join(input_path,"train.csv"))
        
    def create_split(self):
        self.df['kfold'] = -1
        #Shuffling the csv file => to get a new shuffled dataframe
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        #Target value
        y=self.df.target.values
        #Why stratified - because we want the ratio of +ve:-ve samples to be the same
        kf = model_selection.StratifiedKFold(n_splits=self.n_splits)
        
        kfold_df_dict = {}
        
        for fold_, (train_idx, val_idx) in enumerate(kf.split(X=self.df,y=y)):
            df_temp = pd.read_csv(os.path.join(input_path,"train.csv"))
            df_temp['kfold'] = -1
            df_temp['dataset_type'] = 'train'
            df_temp.loc[:,'kfold']=fold_
            df_temp.loc[val_idx,'dataset_type'] = 'val'
            kfold_df_dict[fold_]=df_temp
        
        df_comb_fold = pd.concat(kfold_df_dict[k] for (k,v) in kfold_df_dict.items())
        
        return df_comb_fold


# In[8]:


input_path = "/kaggle/input/siim-isic-melanoma-classification"
num_splits = 10
df_actual_train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
df_kfold = stratification(input_path,num_splits).create_split()


# In[9]:


class model_inceptionv4(torch.nn.Module):
    def __init__(self, pretrained='imagenet'):
        #The super() function is used to give access to methods and properties of a parent or sibling class.
        #The super() function returns an object that represents the parent class.
        #super(Model_ResNext_Pytorch,self).__init__()
        super(model_inceptionv4,self).__init__()
        #self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"]
        #self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        #self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=pretrained)
        self.model = inceptionv4(pretrained=pretrained)
        #self.out = nn.Linear(2048,1)
        self.out = nn.Linear(1536,1)
    
    def forward(self,image,targets):
        # Arguments should match your dataloader arguments wrt dataset being passed
        # in this case it is image, targets
        
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.reshape(bs,-1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.reshape(-1,1).type_as(out)
        )
        # shape and datatype
        return out,loss


# In[10]:


class ClassificationLoader:
    def __init__(self, image_paths, targets, resize, augmentations):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        targets = self.targets[item]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


# In[11]:


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[12]:


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


# In[13]:


class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1
    ):
        
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        tk0 = tqdm(data_loader, total=len(data_loader),disable=False)
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)
                   
            with torch.set_grad_enabled(True):
                loss.backward()
                if (b_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    if b_idx > 0:
                        optimizer.zero_grad()
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            tk0 = tqdm(data_loader, total=len(data_loader), disable=False)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, loss = model(**data)
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions


# In[14]:


def train(fold):
    training_data_path = "/kaggle/input/siic-isic-224x224-images/train"
    df = df_kfold[df_kfold['kfold']==fold]
    device = 'cuda'
    epochs = 50
    train_bs = 32
    val_bs = 16
    
    df_train = df.loc[df['dataset_type']=='train',list(df_actual_train.columns)]
    df_val = df.loc[df['dataset_type']=='val',list(df_actual_train.columns)]
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Resize(299, 299, p=1.0),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    val_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    train_images_list = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path,i + '.png') for i in train_images_list]
    train_targets = df_train.target.values
    
    val_images_list = df_val.image_name.values.tolist()
    val_images = [os.path.join(training_data_path,i + '.png') for i in val_images_list]
    val_targets = df_val.target.values
    
    train_dataset = ClassificationLoader(
        image_paths = train_images,
        targets= train_targets,
        resize = None,
        augmentations = train_aug
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle = False,
        num_workers=4
    )
    
    val_dataset = ClassificationLoader(
        image_paths = val_images,
        targets= val_targets,
        resize = None,
        augmentations = val_aug
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = val_bs,
        shuffle = False,
        num_workers=4
    )
    #Earlier defined class for model
    model = model_inceptionv4(pretrained='imagenet')
    model.to(device)
    
    #Specify an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #Specify an scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode='max'
    )
    # why mode='max' becauase we will be using the metric of AUC
    
    # we would also need early stopping
    es = EarlyStopping(patience=5, mode='max')
    
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device
        )
        predictions, val_loss = Engine.evaluate(
            val_loader,
            model,
            device
        )
        
        predictions = np.vstack((predictions)).ravel()
        # Ravel it because we have only one value
        auc = metrics.roc_auc_score(val_targets, predictions)
        # thats why val_loader shuffle was kept false
        
        scheduler.step(auc)
        print(f"epoch={epoch},auc={auc}")
        # Save it with .bin extension
        model_path = f'model_fold{fold}_epoch{epoch}.bin'
        es(auc, model, model_path)
        if es.early_stop:
            print("Early Stopping")
            break


# In[ ]:


list_fold_pred = []
for fold_ in range(num_splits):
    torch.cuda.empty_cache()
    train(fold=fold_)
    list_fold_pred.append(predict(fold_))

