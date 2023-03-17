class ULTIMUS(nn.Module):
    def __init__(self,neurons=48):
        super(ULTIMUS,self).__init__() 
        self.key = nn.Linear(neurons,8)
        self.query = nn.Linear(neurons,8)
        self.value = nn.Linear(neurons,8)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Linear(8,48) 
        
        
    def forward(self,x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        mul_matrix = torch.matmul(torch.transpose(q,0,1),k)
        am = self.softmax(mul_matrix)/(8**0.5)
        z = torch.matmul(v,am)
        output = self.out(z)
        
        return output
        
class Transformer(nn.Module):
    def __init__(self):
        # To not to refer base class explicity we user super().
        super(Transformer,self).__init__()
        neurons = 48
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=48,kernel_size=3,stride=1)
        
        # 26x26x48
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.ultims1 = ULTIMUS()
        self.ultims2 = ULTIMUS()
        self.ultims3 = ULTIMUS()
        self.ultims4 = ULTIMUS()
        
        self.ffc = nn.Linear(48,10)
        
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
       
        x = x.view(-1,48)
        x = self.ultims1(x)
        x = self.ultims2(x)
        x = self.ultims3(x)
        x = self.ultims4(x)
        x = self.ffc(x) 
        
        
        return x
