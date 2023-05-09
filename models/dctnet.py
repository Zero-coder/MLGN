from distutils.command.config import config
import torch.nn as nn
import math
import numpy as np
import torch
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim = (-d))
        r = torch.stack((t.real, t.imag), -1)
        return r
    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
        return t.real
try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = rfft(v, 1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    

    return V
def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

# class senet_block(nn.Module):
#     def __init__(self, channel=512, ratio=1):
#         super(dct_channel_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, channel // 4, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channel //4, channel, bias=False),
#                 nn.Sigmoid()
#         )

#     def forward(self, x):
#         # b, c, l = x.size() # (B,C,L)
#         # y = self.avg_pool(x) # (B,C,L) -> (B,C,1)
#         # print("y",y.shape)
#         x = x.permute(0,2,1)
#         b, c, l = x.size() 
#         y = self.avg_pool(x).view(b, c) # (B,C,L) ->(B,C,1)
#         # print("y",y.shape)
#         # y = self.fc(y).view(b, c, 96)

#         y = self.fc(y).view(b,c,1)
#         # print("y",y.shape)
#         # return x * y
#         return (x*y).permute(0,2,1)
class dct_channel_block(nn.Module):
    def __init__(self, channel):
        super(dct_channel_block, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool1d(1) #innovation
        self.fc = nn.Sequential(
                nn.Linear(channel, channel*2, bias=False),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear( channel*2, channel, bias=False),
                nn.Sigmoid()
        )
        # self.dct_norm = nn.LayerNorm([512], eps=1e-6)
  
        self.dct_norm = nn.LayerNorm([96], eps=1e-6)#for lstm on length-wise
        # self.dct_norm = nn.LayerNorm([36], eps=1e-6)#for lstm on length-wise on ill with input =36


    def forward(self, x):
        b, c, l = x.size() # (B,C,L) (32,96,512)
        # y = self.avg_pool(x) # (B,C,L) -> (B,C,1)
        
        # y = self.avg_pool(x).view(b, c) # (B,C,L) -> (B,C,1)
        # print("y",y.shape
        # y = self.fc(y).view(b, c, 96)
        list = []
        for i in range(c):
            freq=dct(x[:,i,:])     
            # print("freq-shape:",freq.shape)
            list.append(freq)
         
        

        stack_dct=torch.stack(list,dim=1)
        stack_dct = torch.tensor(stack_dct)
        '''
        for traffic mission:f_weight = self.dct_norm(f_weight.permute(0,2,1))#matters for traffic datasets
        '''
        
        lr_weight = self.dct_norm(stack_dct) 
        lr_weight = self.fc(stack_dct)
        lr_weight = self.dct_norm(lr_weight) 
        
        # print("lr_weight",lr_weight.shape)
        return x *lr_weight #result


if __name__ == '__main__':
    
    tensor = torch.rand(8,7,96)
    dct_model = dct_channel_block()
    result = dct_model.forward(tensor) 
    print("result.shape:",result.shape)


