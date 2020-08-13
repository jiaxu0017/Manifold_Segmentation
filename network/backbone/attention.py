import torch
from torch.nn import Module,Conv2d,Parameter,Softmax

torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']



class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self,x):
        '''
        Calcuate attetion between channels
        Args:
            x: input feature maps (B * C * H * W)

        Returns:
            out: attention value + input feature (B * C * H * W)
            attention: B * C * C

        '''

        m_batchsize, C, height, wight = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize,C, -1).permute(0,2,1)
        proj_value = x.view(m_batchsize,C, -1)

        energy = torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention,proj_value)
        out = out.view(m_batchsize,C, height, wight)
        mean = torch.mean(out)
        out = out/mean

        out = self.gamma*out + x
        return out

class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




if __name__ == '__main__':
    input = torch.ones([1,16,9,5])
    for i  in range(9):
        for j in range(5):
            input[:,:,i,j] = i * 5 + j
    # print(input.size())
    print(input[0,1,])

    # test kronecker
    output_H, output_W , output= kronecker(input)
    # print('H & W:',output_H.size(), output_W.size())
    # print('out',output.size())
    print('H & W:',output_H.shape, output_W.shape)
    # print(output)

    # test Inverse_kronecker
    # out = kronecker(input)
    # print(H[0,1,],W[0,1,])
    out = Inverse_kronecker(output, input.shape[0],input.shape[1],input.shape[2],input.shape[3])
    print(out.shape)
    # # print(out[0,1,])
    # out = out/5


    # test PAM_Module
    # model = PAM_Module(16)
    # out = model(input)
    # print(out.shape)

