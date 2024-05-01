from segment_anything import sam_model_registry
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import timm
from timm.models.resnet import _cfg


class SAM_LST(nn.Module):

    def __init__(self):
        super(SAM_LST, self).__init__()

        self.sam, _ = sam_model_registry["vit_b"](image_size=512,
                                                  num_classes=7,
                                                  checkpoint="/data/shanjuan/home/pretrained_models/pretrained_models/sam_vit_b_01ec64.pth",
                                                  pixel_mean=[0.3394, 0.3598, 0.3226],
                                                  pixel_std=[0.2037, 0.1899, 0.1922])
        for n, p in self.sam.named_parameters():
            p.requires_grad = False

        for n, p in self.sam.named_parameters():

            if "output_upscaling" in n:
                p.requires_grad = True
            if "prompt_generator" in n:
                p.requires_grad = True

        for n, p in self.sam.image_encoder.blocks.named_parameters():
            if "p2t_mlp" in n:
                p.requires_grad = True
            if "p2t_attn" in n:
                p.requires_grad = True
            if "attn_0" in n:
                p.requires_grad = True
            if "attn_1" in n:
                p.requires_grad = True
            if "down_proj" in n:
                p.requires_grad = True
            if "up_proj" in n:
                p.requires_grad = True
            
            if "norm3" in n:
                p.requires_grad = True
            if "norm4" in n:
                p.requires_grad = True
            if "norm5" in n:
                p.requires_grad = True
            if "norm6" in n:
                p.requires_grad = True
            if "MLP_Adapter" in n:
                p.requires_grad = True

            if "Space_Adapter" in n:
                p.requires_grad = True
            if "Depth_Adapter" in n:
                p.requires_grad = True

            if "d_convs" in n:
                p.requires_grad = True


    def forward(self, x, multimask_output = None, image_size =None):

        x = self.sam(x, multimask_output=multimask_output, image_size=image_size)  #, CNN_input=cnn_outs)

        return x


if __name__ == "__main__":

    net = SAM_LST().cuda()
    out = net(torch.rand(1, 3, 512, 512).cuda(), 1, 512)
    parameter = 0
    select = 0
    for n, p in net.named_parameters():

        parameter += len(p.reshape(-1))
        if p.requires_grad == True:
            select += len(p.reshape(-1))
    print(select / parameter * 100)

    print(out['masks'].shape)
