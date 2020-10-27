import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _,_,kH,kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:,:,kH//2,kW//2 + (mask_type == 'B'):] = 0
        self.mask[:,:,kH//2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ConvnetBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        super(ConvnetBlock, self).__init__(*args, **kwargs)
        
    def forward(self, x):

        #
        # Problem 5a: Implement a residual convnet block as described in Lecture 7.
        #             Use a kernel size of 3. Do not implement 1x1 convolutions.
        #
 
        raise NotImplementedError

class MaskedConvnetBlock(nn.Module):
    def __init__(self, filters, *args, **kwargs):
        super(MaskedConvnetBlock, self).__init__(*args, **kwargs)
        
    def forward(self, x):

        #
        # Problem 6a: Implement a masked residual convnet block as described in Lecture 7.
        #             Use a kernel size of 3. Do not implement 1x1 convolutions.
        #             Use the MaskedConv2d to implement a masked convolution.
        #             You'll want to use mask-type B.
        #
 
        raise NotImplementedError

class PixelCNN(nn.Module):
    def __init__(self, capacity=32, depth=9, *args, **kwargs):
        super(PixelCNN, self).__init__(*args, **kwargs)
        self.capacity = capacity

        self.embed = MaskedConv2d('A', 1, capacity, 9, padding=4, bias=False)
        
        self.resnet = nn.ModuleList()
        for i in range(depth): self.resnet.append(MaskedConvnetBlock(capacity))
        
        self.image = MaskedConv2d('B', capacity, 1, 3, padding=1, bias=True)
        self.bias = nn.Parameter(torch.Tensor(28,28))

        for name, parm in self.named_parameters():
            if name.endswith('weight'): nn.init.normal_(parm, 0, .01)
            if name.endswith('bias'): nn.init.constant_(parm, 0.0)
    
    def sample(self, n):
        x = torch.zeros(n,1,28,28).cuda()
        for i in range(28):
            for j in range(28):
                p = torch.sigmoid(self.forward(x).detach()[:,:,i,j])
                x[:,:,i,j] = torch.bernoulli(p)
        
        return x
    
    def forward(self, x):
        zx = F.relu(self.embed(x))
        for layer in self.resnet: zx = layer(zx)
        return self.image(zx) + self.bias[None,None,:,:]

class GaussianVAEDecoder(nn.Module):
    def __init__(self, capacity=32, depth=51, autoregress=False, *args, **kwargs):
        super(GaussianVAEDecoder, self).__init__(*args, **kwargs)
        self.capacity = capacity

        self.embed = nn.Linear(49, capacity*7*7, bias=False)
        
        self.resnet = nn.ModuleList()
        for i in range(depth): self.resnet.append(ConvnetBlock(capacity))
        
        self.image = nn.ConvTranspose2d(capacity, 1, 4, stride=4, bias=True)
        self.bias = nn.Parameter(torch.Tensor(28,28))

        for name, parm in self.named_parameters():
            if name.endswith('weight'): nn.init.normal_(parm, 0, .01)
            if name.endswith('bias'): nn.init.constant_(parm, 0.0)
            
    def sample(self, z, sigma):
        return torch.normal(self(z), sigma).clamp(0,1)

    def forward(self, s):
        zx = F.relu(self.embed(s.view(-1,49)).view(-1,self.capacity,7,7))
        for layer in self.resnet: zx = layer(zx)
        return torch.sigmoid(self.image(zx) + self.bias[None,None,:,:])

class DiscreteVAEDecoder(nn.Module):
    def __init__(self, capacity=32, depth=51, autoregress=False, *args, **kwargs):
        super(DiscreteVAEDecoder, self).__init__(*args, **kwargs)
        self.capacity = capacity

        self.embed = nn.Linear(49, capacity*7*7, bias=False)
        
        self.resnet = nn.ModuleList()
        for i in range(depth): self.resnet.append(ConvnetBlock(capacity))
        
        self.image = nn.ConvTranspose2d(capacity, 1, 4, stride=4, bias=True)
        self.bias = nn.Parameter(torch.Tensor(28,28))

        # regardless whether we autoregress, warm up without it
        self.autoregress = False 
        if autoregress:
            self.pixelcnn = PixelCNN()

        for name, parm in self.named_parameters():
            if name.endswith('weight'): nn.init.normal_(parm, 0, .01)
            if name.endswith('bias'): nn.init.constant_(parm, 0.0)
            
    def sample(self, z):
        if self.autoregress:
            x = torch.zeros(z.shape[0],1,28,28).cuda()
            for i in range(28):
                for j in range(28):
                    p = torch.sigmoid(self.forward(z,x).detach()[:,:,i,j])
                    x[:,:,i,j] = torch.bernoulli(p)
        else:
            p = torch.sigmoid(self(z).detach())
            x = torch.bernoulli(p)

        return x

    def forward(self, s, x=None):
        zx = F.relu(self.embed(s.view(-1,49)).view(-1,self.capacity,7,7))
        for layer in self.resnet: zx = layer(zx)
        xr = self.image(zx) + self.bias[None,None,:,:]
        return xr if not self.autoregress else self.pixelcnn(x) + xr

class IAF(nn.Module):
    def __init__(self, filters, depth=9, *args, **kwargs):
        super(IAF, self).__init__(*args, **kwargs)
        
        self.embedz = MaskedConv2d('A',1,filters,3,1,1,bias=False)
        self.embedh = nn.Conv2d(filters,filters,3,1,1,bias=False)
        
        resnet = []
        for i in range(9): resnet.append(MaskedConvnetBlock(filters))
        self.resnet = nn.Sequential(*resnet)
        self.m = MaskedConv2d('B',filters,1,3,padding=1,bias=True)
        self.s = MaskedConv2d('B',filters,1,3,padding=1,bias=True)
        
    def forward(self, z, h):
        u = F.relu(self.embedz(z) + self.embedh(h))
        u = self.resnet(u)
        return self.m(u), 1 + self.s(u)

class VAEEncoder(nn.Module):
    def __init__(self, capacity=32, depth=9, flows=0, *args, **kwargs):
        super(VAEEncoder, self).__init__(*args, **kwargs)
        self.capacity = capacity

        self.embed = nn.Conv2d(1, capacity, 7, padding=3, stride=4, bias=False)
        
        self.resnet = nn.ModuleList()
        for i in range(depth): self.resnet.append(ConvnetBlock(capacity))
        self.mu = nn.Conv2d(capacity, 1, 3, padding=1, bias=True)
        self.var = nn.Conv2d(capacity, 1, 3, padding=1, bias=True)
        
        self.flows = nn.ModuleList()
        for f in range(flows): self.flows.append(IAF(capacity))

        for name, parm in self.named_parameters():
            if name.endswith('weight'): nn.init.normal_(parm, 0, .01)
            if name.endswith('bias'): nn.init.constant_(parm, 0.0)

    def forward(self, x, epsilon):
        h = F.relu(self.embed(x))
        for layer in self.resnet: h = layer(h)
        mu, logvar = self.mu(h), self.var(h)
        
        z = mu + torch.exp(.5*logvar)*epsilon
        logqzx = 0.5*(logvar + torch.pow((z-mu),2)/logvar.exp()).view(-1,49).sum(1)

 
        for t, flow in enumerate(self.flows):
            #
            # Problem 6d: Calculate an inverse autoregressive flow z.
            #             While calculating z, accumulate a calculation of its probability.
            #

            raise NotImplementedError

        return z.view(-1,49), logqzx, mu.view(-1,49), logvar.view(-1,49)

