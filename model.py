import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url

model_urls ={
     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class ConvBNReLU(nn.Sequential):
    def __init__(self,in_channel,out_channel,kernel_size =3,stride=1,groups=1):
        padding =(kernel_size -1)//2
        super(ConvBNReLU,self).__init__(
            nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding,groups=groups,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self,in_channel,out_channel,stride,expand_ratio):
        super(InvertedResidual,self).__init__()
        self.stride = stride
        assert stride in [1,2]
        hidden_channel = int(round(in_channel * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channel == out_channel

        layers=[]
        if expand_ratio !=1:
            #1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel,hidden_channel,kernel_size=1))
        layers.extend([
            #3x3 depthwise conv
            ConvBNReLU(hidden_channel,hidden_channel,stride=stride,groups=hidden_channel),
            #1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel,out_channel,1,1,0,bias=False),
            nn.BatchNorm2d(out_channel),

        ])
        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=6,width_mult=1.0): #6个分类
        super(MobileNetV2,self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #构建第一层
        input_channel =int(input_channel * width_mult)
        self.last_channel =int(last_channel * width_mult)
        features = [ConvBNReLU(3,input_channel,stride=2)]
        #构建中间层
        for t,c,n,s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride =s if i == 0 else 1
                features.append(block(input_channel,output_channel,stride,expand_ratio=t))
                input_channel = output_channel
        #构建最后层
        features.append(ConvBNReLU(input_channel,self.last_channel,kernel_size=1))
        #组合所有层
        self.features = nn.Sequential(*features)

        #构建分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel,num_classes),
        )

        #初始化权重
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)
        

        #前向传播
    def forward(self,x):
        x = self.features(x)
        x = x.mean([2,3])
        x = self.classifier(x)
        return x
        
def mobilenet_v2(pretrained=False,progress=True,num_classes=6,**kwargs):
    model = MobileNetV2(num_classes,**kwargs)
    if pretrained:
        state_dict =load_state_dict_from_url(model_urls['mobilenet_v2'],progress =progress)
        #删除分类器权重
        if num_classes != 1000: 
            state_dict.pop('classifier.1.weight',None)
            state_dict.pop('classifier.1.bias',None)
        #加载预训练权重
            model_dict = model.state_dict()
            pretrained_dict = {k:v for k,v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Loaded pretrained weights for feature extraction.{len(pretrained_dict)}/{len(state_dict)} layers loaded.")
        else:
            model.load_state_dict(state_dict)
    return model