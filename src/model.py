import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_pad, ConvWithActivation, DeConvWithActivation

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(ResnetBlock, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return F.relu(out)

class Basic_Encoder(nn.Module):
    def __init__(self, n_in_channel=3):
        super(Basic_Encoder, self).__init__()
        self.conv1 = ConvWithActivation(n_in_channel,32,kernel_size=4,stride=2,padding=1)
        self.conva = ConvWithActivation(32,32,kernel_size=3, stride=1, padding=1)
        self.convb = ConvWithActivation(32,64, kernel_size=4, stride=2, padding=1)
        self.res1 = ResnetBlock(64,64)
        self.res2 = ResnetBlock(64,64)
        self.res3 = ResnetBlock(64,128,same_shape=False)
        self.res4 = ResnetBlock(128,128)
        self.res5 = ResnetBlock(128,256,same_shape=False)
        self.res6 = ResnetBlock(256,256)
        self.res7 = ResnetBlock(256,512,same_shape=False)
        self.res8 = ResnetBlock(512,512)
        self.conv2 = ConvWithActivation(512,512,kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
        x = self.convb(x)
        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        x = self.res3(x)
        con_x3 = x
        x = self.res4(x)
        x = self.res5(x)
        con_x4 = x
        x = self.res6(x)
        x_mask=x
        x = self.res7(x)
        x = self.res8(x)
        x = self.conv2(x)
        return x, x_mask, con_x1, con_x2, con_x3, con_x4

class Basic_Decoder(nn.Module):
    def __init__(self,):
        super(Basic_Decoder, self).__init__()
        self.deconv1 = DeConvWithActivation(512,256,kernel_size=3,padding=1,stride=2)
        self.deconv2 = DeConvWithActivation(256*2,128,kernel_size=3,padding=1,stride=2)
        self.deconv3 = DeConvWithActivation(128*2,64,kernel_size=3,padding=1,stride=2)
        self.deconv4 = DeConvWithActivation(64*2,32,kernel_size=3,padding=1,stride=2)
        self.deconv5 = DeConvWithActivation(64,3,kernel_size=3,padding=1,stride=2)

        self.lateral_connection1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 256, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 128, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 64, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0,stride=1),)

        self.conv_o1 = nn.Conv2d(64,3,kernel_size=1)
        self.conv_o2 = nn.Conv2d(32,3,kernel_size=1)

    def forward(self, input, con_x1, con_x2, con_x3, con_x4):
        x = self.deconv1(input)
        x = torch.cat([self.lateral_connection1(con_x4), x], dim=1)
        x = self.deconv2(x)
        x = torch.cat([self.lateral_connection2(con_x3), x], dim=1)
        x = self.deconv3(x)
        xo1 = x
        x = torch.cat([self.lateral_connection3(con_x2), x], dim=1)
        x = self.deconv4(x)
        xo2 = x
        x = torch.cat([self.lateral_connection4(con_x1), x], dim=1)
        x = self.deconv5(x)
        x_o1 = self.conv_o1(xo1)
        x_o2 = self.conv_o2(xo2)
        initial_result = x
        return x_o1, x_o2, initial_result

class Stroke_Decoder(nn.Module):
    def __init__(self,):
        super(Stroke_Decoder, self).__init__()
        self.mask_deconv_a = DeConvWithActivation(512,256,kernel_size=3,padding=1,stride=2)
        self.mask_conv_a = ConvWithActivation(256,128,kernel_size=3,padding=1,stride=1)
        self.mask_deconv_b = DeConvWithActivation(256,128,kernel_size=3,padding=1,stride=2)
        self.mask_conv_b = ConvWithActivation(128,64,kernel_size=3,padding=1,stride=1)
        self.mask_deconv_c = DeConvWithActivation(128,64,kernel_size=3,padding=1,stride=2)
        self.mask_conv_c = ConvWithActivation(64,32,kernel_size=3,padding=1,stride=1)
        self.mask_deconv_d = DeConvWithActivation(64,32,kernel_size=3,padding=1,stride=2)
        self.mask_conv_d = nn.Conv2d(32,3,kernel_size=1)
    def forward(self, x_mask, con_x1, con_x2, con_x3, con_x4):
        mm = self.mask_deconv_a(torch.cat([x_mask,con_x4],dim=1))
        mm = self.mask_conv_a(mm)
        mm = self.mask_deconv_b(torch.cat([mm,con_x3],dim=1))
        mm = self.mask_conv_b(mm)
        mm = self.mask_deconv_c(torch.cat([mm,con_x2],dim=1))
        mm = self.mask_conv_c(mm)
        mm = self.mask_deconv_d(torch.cat([mm,con_x1],dim=1))
        mm = self.mask_conv_d(mm)
        return mm

class Refinementwork(nn.Module):
    def __init__(self, n_in_channel=6):
        super(Refinementwork, self).__init__()
        cnum = 32
        ####downsapmle
        self.coarse_conva = ConvWithActivation(n_in_channel, cnum, kernel_size=5, stride=1, padding=2)
        self.coarse_convb = ConvWithActivation(cnum, 2*cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_convc = ConvWithActivation(2*cnum, 2*cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convd = ConvWithActivation(2*cnum, 4*cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_conve = ConvWithActivation(4*cnum, 4*cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convf = ConvWithActivation(4*cnum, 4*cnum, kernel_size=3, stride=1, padding=1)
        ### astrous
        self.astrous_net = nn.Sequential(
            ConvWithActivation(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            ConvWithActivation(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            ConvWithActivation(4*cnum, 4*cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            ConvWithActivation(4*cnum, 4*cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
        )
        ###astrous
        ### upsample
        self.coarse_convk = ConvWithActivation(4*cnum, 4*cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convl = ConvWithActivation(4*cnum, 4*cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_deconva = DeConvWithActivation(4*cnum*3, 2*cnum, kernel_size=3,padding=1,stride=2)
        self.coarse_convm = ConvWithActivation(2*cnum, 2*cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = DeConvWithActivation(2*cnum*3, cnum, kernel_size=3,padding=1,stride=2)
        self.coarse_convn = nn.Sequential(
            ConvWithActivation(cnum, cnum//2, kernel_size=3, stride=1, padding=1),
            ConvWithActivation(cnum//2, 3, kernel_size=3, stride=1, padding=1, activation=None),
        )   
        self.c1 = nn.Conv2d(32,64,kernel_size=1)    
        self.c2 = nn.Conv2d(64,128,kernel_size=1)

    def forward(self, initial_result, con_x1, con_x2, masks=None):
        # x = self.coarse_conva(initial_result)
        x = self.coarse_conva(torch.cat([masks, initial_result], dim=1))
        x = self.coarse_convb(x)
        x = self.coarse_convc(x)
        x_c1 = x     ###concate feature1
        x = self.coarse_convd(x)
        x = self.coarse_conve(x)
        x = self.coarse_convf(x)
        x_c2 = x    ###concate feature2
        x = self.astrous_net(x)
        x = self.coarse_convk(x)
        x = self.coarse_convl(x)
        x = self.coarse_deconva(torch.cat([x, x_c2,self.c2(con_x2)],dim=1))
        x = self.coarse_convm(x)
        x = self.coarse_deconvb(torch.cat([x,x_c1,self.c1(con_x1)],dim=1))
        refinement_result = self.coarse_convn(x)
        return refinement_result

class SGNet(nn.Module):
    def __init__(self, n_in_channel=3):
        super(SGNet, self).__init__()
        self.basic_encoder = Basic_Encoder(n_in_channel)
        self.basic_decoder = Basic_Decoder()
        self.stroke_decoder = Stroke_Decoder()
        self.refinement = Refinementwork(n_in_channel+3)

    def forward(self, input_image, masks, training=True):
        x, x_mask, con_x1, con_x2, con_x3, con_x4 = self.basic_encoder(input_image)
        x_o1, x_o2, initial_result = self.basic_decoder(x, con_x1, con_x2, con_x3, con_x4)
        stroke = self.stroke_decoder(x_mask, con_x1, con_x2, con_x3, con_x4)
        # refinement_result = self.refinement(initial_result, con_x1, con_x2)
        if training:
            refinement_result = self.refinement(initial_result, con_x1, con_x2, masks)
        else:
            stroke = stroke.mul(1).add_(0.5).clamp_(0, 1)
            refinement_result = self.refinement(initial_result, con_x1, con_x2, (1-stroke))
        return x_o1, x_o2, initial_result, refinement_result, stroke