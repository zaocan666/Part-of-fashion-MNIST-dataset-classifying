import torch
import torch.nn as nn
import torch.nn.functional as F

#based on https://github.com/39239580/googlenet-pytorch
class BasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn=nn.BatchNorm2d(out_channels,eps=0.001)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return F.relu(x,inplace=True)
    
# incption
class InceptionModule(nn.Module):
    def __init__(self,in_channels,conv1x1,reduce3x3,conv3x3,reduce5x5,conv5x5,pool_features):
        super().__init__()
        self.branch1x1=BasicConv2d(in_channels,conv1x1,kernel_size=1,stride=1,padding=0)
        
        self.branch3x3_1=BasicConv2d(in_channels,reduce3x3,kernel_size=1,stride=1,padding=0)
        self.branch3x3_2=BasicConv2d(reduce3x3,conv3x3,kernel_size=3,padding=1,stride=1)
        
        self.branch5x5_1=BasicConv2d(in_channels,reduce5x5,kernel_size=1,stride=1,padding=0)
        self.branch5x5_2=BasicConv2d(reduce5x5,conv5x5,kernel_size=5,stride=1,padding=2)
       
        self.branch_pool=BasicConv2d(in_channels,pool_features,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        branch1x1=self.branch1x1(x)
        
        branch3x3_a=self.branch3x3_1(x)
        branch3x3_b=self.branch3x3_2(branch3x3_a)
        
        branch5x5_a=self.branch5x5_1(x)
        branch5x5_b=self.branch5x5_2(branch5x5_a)

        branch_pool=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)
        
        outputs=[branch1x1,branch3x3_b,branch5x5_b,branch_pool]
        return torch.cat(outputs,1)
    #torch.cat(inputs,1) ?inputs???????????????
    #torch.cat(inputs,0) ?inputs???????????

class InceptionAux(nn.Module):
    def __init__(self,in_channels,out_channels,out,num_classes):
        super().__init()
        self.conv0=BasicConv2d(in_channels,out_channels,kernel_size=1,stride=1)
        self.conv1.stddev=0.01
        self.fc1=nn.Linear(out_channels,out)
        self.fc1.stddev=0.001
        self.fc2=nn.Linear(out,num_classes)
        self.fc2.stddev=0.001
        
    def forward(self,x):
        x=F.avg_pool2d(x,kernel_size=5,stride=3)
        x=self.conv0(x)
        x=x.view(x.size(0),-1) # ???????????
        x=self.fc1(x)
        x=self.fc2(x)
        return x


class googleNet(nn.Module):
    def __init__(self,num_classes=10):#,aux_logits_state):
        super().__init__()
        num_class=num_classes
#        self.aux_logits=aux_logits_state
#        self.transform_input=transform_input #??  28 *28 *1
        self.Conv2d1=BasicConv2d(1,8,kernel_size=5,stride=1,padding=2)#?? 28*28*8
        self.Conv2d2=BasicConv2d(8,32,kernel_size=3,stride=1,padding=1)# ?? 28*28*32

        self.Mixed_3a=InceptionModule(32,16,24,32,4,8,pool_features=8) #?? 28*28*64
        self.Mixed_3b=InceptionModule(64,32,32,48,8,24,pool_features=16) #?? 28*28*120

        self.Mixed_4a=InceptionModule(120,48,24,52,4,12,pool_features=16) #?? 14*14*128
        self.Mixed_4b=InceptionModule(128,40,28,56,6,16,pool_features=16) #?? 14*14*128
        self.Mixed_4c=InceptionModule(128,32,32,64,12,16,pool_features=16) #?? 14*14*128
        self.Mixed_4d=InceptionModule(128,28,36,72,8,16,pool_features=16)  #?? 14*14*132
        self.Mixed_4e=InceptionModule(132,64,40,80,8,32,pool_features=32) #?? 14*14*208
        
        self.Mixed_5a=InceptionModule(208,64,40,80,8,32,pool_features=32) #?? 7*7*208
        self.Mixed_5b=InceptionModule(208,96,48,96,12,32,pool_features=32) ##?? 7*7*256

        self.dropout_layer = nn.Dropout(0.5)

        #self.fc1=nn.Linear(256,80)  
        self.fc2=nn.Linear(256,num_class)
    def forward(self,transform_input):
        x=transform_input
        # 28*28*1
        x = self.Conv2d1(x) 
        x = self.Conv2d2(x)       
        # 28*28*32
        x = self.Mixed_3a(x)
        x = self.Mixed_3b(x)
        #28*28*120
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)#??14*14*120       
        x = self.Mixed_4a(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        #14*14*208
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)#??7*7*208        
        x = self.Mixed_5a(x)
        x = self.Mixed_5b(x)  
        # 7*7*256       
        x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)#??1*1*256
        x = self.dropout_layer(x) #??1*1*256
        x = x.view(x.size(0), -1) #256
        #x = self.fc1(x)
        x = self.fc2(x)
        return x
