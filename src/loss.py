import torch
from torch import nn
import torch.nn.functional as F
from src.discriminator import Discriminator_STE

def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def dice_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    
    input = input 
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

class Loss(nn.Module):
    def __init__(self, extractor, lr, betasInit=(0.5, 0.9)):
        super(Loss, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = Discriminator_STE(3)
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()

    def forward(self, input, mask, x_o1,x_o2,x_o3,output,mm, gt):
        self.discriminator.zero_grad()
        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(output, mask)
        D_fake = D_fake.mean().sum() * 1
        D_loss = torch.mean(F.relu(1.+D_real)) + torch.mean(F.relu(1.+D_fake))
        D_fake = -torch.mean(D_fake)

        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()
        
        output_comp = mask * input + (1 - mask) * output
        holeLoss = 10 * self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = 2*self.l1(mask * output, mask * gt)

        mask_loss = dice_loss(mm, 1-mask)
        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        imgs1 = F.interpolate(gt, scale_factor=0.25)
        imgs2 = F.interpolate(gt, scale_factor=0.5)
        msrloss = 8 * self.l1((1-mask)*x_o3,(1-mask)*gt) + 0.8*self.l1(mask*x_o3, mask*gt)+\
                    6 * self.l1((1-masks_b)*x_o2,(1-masks_b)*imgs2)+1*self.l1(masks_b*x_o2,masks_b*imgs2)+\
                    5 * self.l1((1-masks_a)*x_o1,(1-masks_a)*imgs1)+0.8*self.l1(masks_a*x_o1,masks_a*imgs1)

        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

        GLoss = msrloss + holeLoss + validAreaLoss+ prcLoss + styleLoss + 0.1 * D_fake + mask_loss  
        return GLoss.sum()