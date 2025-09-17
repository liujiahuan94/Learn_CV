"""
Copyright (c) 2025 liujiahuan. All Rights Reserved.
Contact me by email: 740988959@qq.com

基于pytorch框架的一种VGG模型实现.
"""


import torch
import torch.nn as nn


class VGG(nn.Module):
    '''
    VGG模型定义类
    '''
    def __init__(self, model_cfg, num_classes):
        '''
        VGG类初始化函数

        Args:
            model_cfg(str): 模型配置参数
            num_classes(int): 模型输出类别数

        '''
        super().__init__()
        self.features = self.__make_layers(model_cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def __make_layers(self, cfg):
        '''
        根据VGG配置搭建模型层

        Args:
            cfg(str): 模型配置信息列表
        Returns:
            模型堆叠层结构
        '''

        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


cfgs = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


if __name__ == '__main__':

    vgg_model = VGG(cfgs['VGG16'], 10)
    x_tensor = torch.randn(1, 3, 224, 224)
    y = vgg_model(x_tensor)
    print(y.size())



