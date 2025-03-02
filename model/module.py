import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from layer import conv3x3, SPA_SMaxPool
from utils.rotation_utils import Ortho6d2Mat

from kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample
from lib.pointnet2.pointnet2_utils import three_nn, three_interpolate
from lib.pointnet2.pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule

class KPConvFPN(nn.Module):
    

    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)
        self.encoding1 = local_fuse(init_dim * 2)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoding2 = local_fuse(init_dim * 4)

        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoding3 = local_fuse(init_dim * 8)

        self.encoder4_1 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoding4 = local_fuse(init_dim * 16)


        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm) 
        self.decoder2 = UnaryBlock(init_dim * 12, init_dim * 4, group_norm) 
        self.decoder1 = LastUnaryBlock(init_dim * 6, init_dim * 2)          

    def forward(self, feats, data_dict, i):
       #import pdb;pdb.set_trace()
        feats_list = []

        points_list = data_dict['pcd_points'][i]
        neighbors_list = data_dict['neighbors'][i]
        subsampling_list = data_dict['subsampling'][i]
        upsampling_list = data_dict['upsampling'][i]

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0]) # B, N, 128
        
        # 3d coords of the points [B,3,N]; features of the points [B,C,N]
        coords1 = points_list[0].unsqueeze(0).permute(0, 2, 1)
        feats_s1 = feats_s1.unsqueeze(0).permute(0, 2, 1)
        feats_s1 = self.encoding1(coords1, feats_s1, 20) # B, 128, N
        feats_s1 = feats_s1.permute(0, 2, 1).squeeze(0)            # B, N, 128
        
        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1]) # B, N, 256
        
        n, c = points_list[1].shape
        if n >=20:
            n = 20
        else:
            n = n
        coords2 = points_list[1].unsqueeze(0).permute(0, 2, 1)
        feats_s2 = feats_s2.unsqueeze(0).permute(0, 2, 1)
        feats_s2 = self.encoding2(coords2, feats_s2, n) 
        feats_s2 = feats_s2.permute(0, 2, 1).squeeze(0)              
        
        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2]) # B, N, 512
        
        n, c = points_list[2].shape
        if n >=10:
            n = 10
        else:
            n = n
        coords3 = points_list[2].unsqueeze(0).permute(0, 2, 1)
        feats_s3 = feats_s3.unsqueeze(0).permute(0, 2, 1)
        feats_s3 = self.encoding3(coords3, feats_s3, n)
        feats_s3 = feats_s3.permute(0, 2, 1).squeeze(0)   
        
        #print(feats_s3.shape)
        #print(feats_s3)
        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3]) # B, N, 1024
        
        n, c = points_list[3].shape
        if n >=10:
            n = 10
        else:
            n = n
        coords4 = points_list[3].unsqueeze(0).permute(0, 2, 1)
        feats_s4 = feats_s4.unsqueeze(0).permute(0, 2, 1)
        feats_s4 = self.encoding4(coords4, feats_s4, n)
        feats_s4 = feats_s4.permute(0, 2, 1).squeeze(0)   
        
        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        latent_s1 = nearest_upsample(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)

        feats_list.append(latent_s1)    # [s4,s3,s2,s1]
        feats_list.reverse()            # [s1,s2,s3,s4]

        return feats_list


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_neighbors(x, feature, k=20, idx=None):
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size*num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    neighbor_x = torch.cat((neighbor_x-x, x), dim=3).permute(0, 3, 1, 2)

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size*num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims) 
    feature = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    neighbor_feat = torch.cat((neighbor_feat-feature, feature), dim=3).permute(0, 3, 1, 2)
  
    return neighbor_x, neighbor_feat


class Mish(nn.Module):
    '''activation function'''
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx):
        ctx = ctx * (torch.tanh(F.softplus(ctx)))
        return ctx

    @staticmethod
    def backward(ctx, grad_output):
        input_grad = (torch.exp(ctx) * (4 * (ctx + 1) + 4 * torch.exp(2 * ctx) + torch.exp(3 * ctx) +
                                        torch.exp(ctx) * (4 * ctx + 6)))/(2 * torch.exp(ctx) + torch.exp(2 * ctx) + 2)
        return input_grad


class local_fuse(nn.Module):
    def __init__(self, input_features_dim):
        super(local_fuse, self).__init__()

        self.mish = Mish()

        self.conv_mlp1 = nn.Conv2d(6, input_features_dim // 2, 1)
        self.bn_mlp1 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_mlp2 = nn.Conv2d(input_features_dim*2, input_features_dim // 2, 1)
        self.bn_mlp2 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_down1 = nn.Conv1d(input_features_dim, input_features_dim // 8, 1, bias=False)
        self.conv_down2 = nn.Conv1d(input_features_dim, input_features_dim // 8, 1, bias=False)

        self.conv_up = nn.Conv1d(input_features_dim // 8, input_features_dim, 1)
        self.bn_up = nn.BatchNorm1d(input_features_dim)

    def forward(self, xyz, features, k):
        # Local Context fusion
        neighbor_xyz, neighbor_feat = get_neighbors(xyz, features, k=k)
        neighbor_xyz = F.relu(self.bn_mlp1(self.conv_mlp1(neighbor_xyz))) # B,C/2,N,k
        neighbor_feat = F.relu(self.bn_mlp2(self.conv_mlp2(neighbor_feat))) 
        f_encoding = torch.cat((neighbor_xyz, neighbor_feat), dim=1) # B,C,N,k
        f_encoding = f_encoding.max(dim=-1, keepdim=False)[0] # B,C,N
        f_encoding_1 = F.relu(self.conv_down1(f_encoding)) # B,C/8,N
        f_encoding_2 = F.relu(self.conv_down2(f_encoding)) 
        f_encoding_channel = f_encoding_1.mean(dim=-1, keepdim=True)[0] # B,C/8,1
        f_encoding_space = f_encoding_2.mean(dim=1, keepdim=True)[0] # B,1,N
        final_encoding = torch.matmul(f_encoding_channel, f_encoding_space) # B,C/8,N
        final_encoding = torch.sqrt(final_encoding+1e-12) 
        final_encoding = final_encoding + f_encoding_1 + f_encoding_2
        final_encoding = F.relu(self.bn_up(self.conv_up(final_encoding))) # B,C,N
        f_local = f_encoding-final_encoding   
        # Activation
        f_local = self.mish(f_local)

        return f_local
    

class PointNet2MSG(nn.Module):
    def __init__(self, radii_list, dim_in=6, use_xyz=True):
        super(PointNet2MSG, self).__init__()
        self.SA_modules = nn.ModuleList()
        c_in = dim_in
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=radii_list[0],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 16, 16, 32]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_0 = 32 + 32

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=radii_list[1],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_1 = 64 + 64

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=radii_list[2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_2 = 128 + 128

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=radii_list[3],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 128, 256], [c_in, 128, 128, 256]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_3 = 256 + 256

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + dim_in, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=True))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        _, N, _ = pointcloud.size()

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, type='spa_sconv'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation, type=type)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation, type=type)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, type='spa_sconv'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=1, type=type)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 dim_in=1,
                 layers=(3, 4, 23, 3),
                 type='spa_sconv'
                 ):

        self.current_stride = 4
        self.current_dilation = 1
        self.output_stride = 32

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(dim_in, 64, stride=1, type=type)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if type == 'spa_sconv':
            self.maxpool = SPA_SMaxPool(kernel_size=3, stride=2)
        else:
            self.maxpool = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(block, 64, layers[0], type=type) # 32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, type=type) # 16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=2, type=type) # 8
        self.layer4 = self._make_layer(block, 256, layers[3], stride=1, dilation=4, type=type)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, type='spa_sconv'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation, type=type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation, type=type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x5


class FPN(nn.Module):
    def __init__(self, dim_in=[64,128,256], out_dim=256, mode='nearest', align_corners=True, type='spa_sconv', ds_rate=2):
        super(FPN, self).__init__()
        self.ds_rate = ds_rate
        self.conv1 = conv3x3(dim_in[0], out_dim, stride=1, type=type)
        self.bn1 = nn.BatchNorm2d(out_dim)

        self.conv2 = conv3x3(dim_in[1], out_dim, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv3 = conv3x3(dim_in[2], out_dim, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.relu = nn.ReLU(inplace=True)

        if mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x1, x2, x3):

        x3 = self.up(x3)
        x2 = self.bn2(self.conv2(x2))
        x2 = self.relu(x2+x3)

        if self.ds_rate == 4:
            return x2
        else:
            x2 = self.up(x2)
            x1 = self.bn1(self.conv1(x1))
            x1 = self.relu(x1+x2)

            x1 = self.relu(self.bn3(self.conv3(x1)))

            if self.ds_rate == 1:
                x1 = self.up(x1)

            return x1




class FPN_Adaptive(nn.Module):
    def __init__(self, dim_in=[64,128,256], out_dim=256, mode='nearest', align_corners=True, type='spa_sconv', ds_rate=2):
        super(FPN_Adaptive, self).__init__()
        self.ds_rate = ds_rate
        self.conv1 = conv3x3(dim_in[0], out_dim, stride=1, type=type)
        self.bn1 = nn.BatchNorm2d(out_dim)

        self.conv2 = conv3x3(dim_in[1], out_dim, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv3 = conv3x3(dim_in[2], out_dim, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(out_dim)

        self.conv4 = conv3x3(out_dim, out_dim, stride=1, type=type)
        self.bn4 = nn.BatchNorm2d(out_dim)

        self.relu = nn.ReLU(inplace=True)

        if mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x1, x2, x3):

        x3 = self.up(x3)
        x3 = self.bn3(self.conv3(x3))
        x2 = self.bn2(self.conv2(x2))
        x2 = self.relu(x2+x3)

        if self.ds_rate == 4:
            return x2
        else:
            x2 = self.up(x2)
            x1 = self.bn1(self.conv1(x1))
            x1 = self.relu(x1+x2)

            x1 = self.relu(self.bn4(self.conv4(x1)))

            if self.ds_rate == 1:
                x1 = self.up(x1)

            return x1
  

class SphericalFPN(nn.Module):
    def __init__(self, dim_in1=1, dim_in2=3, type='spa_sconv', ds_rate=2):
        super(SphericalFPN, self).__init__()
        self.ds_rate = ds_rate
        assert ds_rate in [1,2,4]
        self.encoder1 = ResNet(BasicBlock, dim_in1, [2, 2, 2, 2], type=type)
        self.encoder2 = ResNet(BasicBlock, dim_in2, [2, 2, 2, 2], type=type)
        self.FPN = FPN(dim_in=[128,256,256], mode='bilinear', type=type, ds_rate=ds_rate)

        if ds_rate in [1,2]:
            self.conv1 = conv3x3(64*2, 128, stride=1, type=type)
            self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128*2, 256, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = conv3x3(256*2, 256, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        y11,y21,y31 = self.encoder1(x1)
        y12,y22,y32 = self.encoder2(x2)
        # import pdb;pdb.set_trace()
        if self.ds_rate in [1,2]:
            y1 = self.relu(self.bn1(self.conv1(torch.cat([y11,y12],1))))
        else:
            y1 = None
        y2 = self.relu(self.bn2(self.conv2(torch.cat([y21,y22],1))))
        y3 = self.relu(self.bn3(self.conv3(torch.cat([y31,y32],1))))
        y = self.FPN(y1,y2,y3)
        return y


class SphericalFPN3(nn.Module):
    def __init__(self, dim_in1=1, dim_in2 = 3, dim_in3 = 384,type='spa_sconv', ds_rate=2):
        super(SphericalFPN3, self).__init__()
        self.ds_rate = ds_rate
        assert ds_rate in [1,2,4]
        self.encoder1 = ResNet(BasicBlock, dim_in1, [2, 2, 2, 2], type=type)
        self.encoder2 = ResNet(BasicBlock, dim_in2, [2, 2, 2, 2], type=type)
        self.encoder3 = ResNet(BasicBlock, dim_in3, [2, 2, 2, 2], type=type)
        
        self.FPN = FPN(dim_in=[128,256,256], mode='bilinear', type=type, ds_rate=ds_rate)

        if ds_rate in [1,2]:
            self.conv1 = conv3x3(64*3, 128, stride=1, type=type)
            self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128*3, 256, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = conv3x3(256*3, 256, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3):
        y11,y21,y31 = self.encoder1(x1)
        y12,y22,y32 = self.encoder2(x2)
        y13,y23,y33 = self.encoder3(x3) 
        # import pdb;pdb.set_trace()
        if self.ds_rate in [1,2]:
            y1 = self.relu(self.bn1(self.conv1(torch.cat([y11,y12, y13],1))))
        else:
            y1 = None
        y2 = self.relu(self.bn2(self.conv2(torch.cat([y21,y22, y23],1))))
        y3 = self.relu(self.bn3(self.conv3(torch.cat([y31,y32, y33],1))))
        y = self.FPN(y1,y2,y3)
        return y

class SphericalFPN4(nn.Module):
    def __init__(self, dim_in1=1, dim_in2 = 3, dim_in3 = 384, dim_in4 = 4,type='spa_sconv', ds_rate=2):
        super(SphericalFPN4, self).__init__()
        self.ds_rate = ds_rate
        assert ds_rate in [1,2,4]
        self.encoder1 = ResNet(BasicBlock, dim_in1, [2, 2, 2, 2], type=type)
        self.encoder2 = ResNet(BasicBlock, dim_in2, [2, 2, 2, 2], type=type)
        self.encoder3 = ResNet(BasicBlock, dim_in3, [2, 2, 2, 2], type=type)
        self.encoder4 = ResNet(BasicBlock, dim_in4, [2, 2, 2, 2], type=type)
        
        self.FPN = FPN(dim_in=[128,256,256], mode='bilinear', type=type, ds_rate=ds_rate)

        if ds_rate in [1,2]:
            self.conv1 = conv3x3(64*4, 128, stride=1, type=type)
            self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128*4, 256, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = conv3x3(256*4, 256, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):
        y11,y21,y31 = self.encoder1(x1)
        y12,y22,y32 = self.encoder2(x2)
        y13,y23,y33 = self.encoder3(x3) 
        y14,y24,y34 = self.encoder4(x4)
        # import pdb;pdb.set_trace()
        if self.ds_rate in [1,2]:
            y1 = self.relu(self.bn1(self.conv1(torch.cat([y11,y12, y13, y14],1))))
        else:
            y1 = None
        y2 = self.relu(self.bn2(self.conv2(torch.cat([y21,y22, y23, y24],1))))
        y3 = self.relu(self.bn3(self.conv3(torch.cat([y31,y32, y33, y34],1))))
        y = self.FPN(y1,y2,y3)
        return y



class SphericalFPNL(nn.Module):
    def __init__(self, dim_in1=1, dim_in2_list=[3], type='spa_sconv', ds_rate=2):
        super(SphericalFPNL, self).__init__()
        self.ds_rate = ds_rate
        assert ds_rate in [1,2,4]
        self.encoder1 = ResNet(BasicBlock, dim_in1, [2, 2, 2, 2], type=type)
        self.encoder2 = []
        for dim_in2 in dim_in2_list:
            self.encoder2.append(ResNet(BasicBlock, dim_in2, [2, 2, 2, 2], type=type).cuda())
        self.FPN = FPN(dim_in=[128,256,256], mode='bilinear', type=type, ds_rate=ds_rate)

        if ds_rate in [1,2]:
            self.conv1 = conv3x3(64*(1+len(dim_in2_list)), 128, stride=1, type=type)
            self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128*(1+len(dim_in2_list)), 256, stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = conv3x3(256*(1+len(dim_in2_list)), 256, stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2_list):
        y11,y21,y31 = self.encoder1(x1)
        y12_list = []
        y22_list = []
        y32_list = []
        for x2, encoder2 in zip(x2_list, self.encoder2):
            y12,y22,y32 = encoder2(x2)
            y12_list.append(y12)
            y22_list.append(y22)
            y32_list.append(y32)
        y12 = torch.cat(y12_list,1)
        y22 = torch.cat(y22_list,1)
        y32 = torch.cat(y32_list,1)
        # import pdb;pdb.set_trace()
        if self.ds_rate in [1,2]:
            y1 = self.relu(self.bn1(self.conv1(torch.cat([y11,y12],1))))
        else:
            y1 = None
        y2 = self.relu(self.bn2(self.conv2(torch.cat([y21,y22],1))))
        y3 = self.relu(self.bn3(self.conv3(torch.cat([y31,y32],1))))
        y = self.FPN(y1,y2,y3)
        return y



class SphericalFPNL_Adaptive(nn.Module):
    def __init__(self, dim_in1=1, dim_in2_list=[3], type='spa_sconv', ds_rate=2):
        super(SphericalFPNL_Adaptive, self).__init__()
        self.ds_rate = ds_rate
        assert ds_rate in [1,2,4]
        self.encoder1 = ResNet(BasicBlock, dim_in1, [2, 2, 2, 2], type=type)
        self.encoder2 = []
        for dim_in2 in dim_in2_list:
            self.encoder2.append(ResNet(BasicBlock, dim_in2, [2, 2, 2, 2], type=type).cuda())
        self.FPN = FPN_Adaptive(dim_in=[64*(1+len(dim_in2_list)),128*(1+len(dim_in2_list)),256*(1+len(dim_in2_list))], mode='bilinear', type=type, ds_rate=ds_rate)

        if ds_rate in [1,2]:
            self.conv1 = conv3x3(64*(1+len(dim_in2_list)), 64*(1+len(dim_in2_list)), stride=1, type=type)
            self.bn1 = nn.BatchNorm2d(64*(1+len(dim_in2_list)))
        self.conv2 = conv3x3(128*(1+len(dim_in2_list)), 128*(1+len(dim_in2_list)), stride=1, type=type)
        self.bn2 = nn.BatchNorm2d(128*(1+len(dim_in2_list)))
        self.conv3 = conv3x3(256*(1+len(dim_in2_list)), 256*(1+len(dim_in2_list)), stride=1, type=type)
        self.bn3 = nn.BatchNorm2d(256*(1+len(dim_in2_list)))
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2_list):
        y11,y21,y31 = self.encoder1(x1)
        y12_list = []
        y22_list = []
        y32_list = []
        for x2, encoder2 in zip(x2_list, self.encoder2):
            y12,y22,y32 = encoder2(x2)
            y12_list.append(y12)
            y22_list.append(y22)
            y32_list.append(y32)
        y12 = torch.cat(y12_list,1)
        y22 = torch.cat(y22_list,1)
        y32 = torch.cat(y32_list,1)
        if self.ds_rate in [1,2]:
            y1 = self.relu(self.bn1(self.conv1(torch.cat([y11,y12],1))))
        else:
            y1 = None
        y2 = self.relu(self.bn2(self.conv2(torch.cat([y21,y22],1))))
        y3 = self.relu(self.bn3(self.conv3(torch.cat([y31,y32],1))))
        y = self.FPN(y1,y2,y3)
        return y

class V_Branch(nn.Module):
    def __init__(self, in_dim=256, ncls=1, resolution=32):
        super(V_Branch, self).__init__()
        self.ncls = ncls
        self.res = resolution
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.rho_classifier = nn.Sequential(
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, ncls, 1),
        )

        self.phi_classifier = nn.Sequential(
            nn.Conv1d(1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, ncls, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, l, cls=None):
        
        emb = self.mlp(x)

        rho_emb = torch.max(emb, dim=2)[0]
        rho_prob = self.rho_classifier(rho_emb)

        phi_emb = torch.max(emb, dim=3)[0]
        phi_prob = self.phi_classifier(phi_emb)

        if self.ncls > 1:
            b,_,n = rho_prob.size()
            index = cls.reshape(b,1,1).expand(b,1,n)

            rho_prob = torch.gather(rho_prob, 1, index)
            phi_prob = torch.gather(phi_prob, 1, index)

        rho_prob = rho_prob.squeeze(1)
        phi_prob = phi_prob.squeeze(1)
        vp_rot = self._get_vp_rotation(rho_prob, phi_prob, l).detach()
        return vp_rot, rho_prob, phi_prob
    
    def _get_vp_rotation(self, rho_prob, phi_prob, l):
        b = rho_prob.size(0)
        n = self.res
        assert n == rho_prob.size(1)
        assert n == phi_prob.size(1)
        # if torch.isnan(rho_prob).sum()>0:
        #     import pdb;pdb.set_trace()

        if self.training and 'rho_label' in l.keys():
            rho_label = l['rho_label'].reshape(b).long()
            rho_noise = (torch.rand(b)>0.5).long()*torch.randint(-3,3,(b,)).long()
            rho_label = rho_label + rho_noise.to(rho_label.device)
            rho_label = torch.clamp(rho_label, 0, n-1)

        else:
            rho_label = torch.sigmoid(rho_prob)
            rho_label = torch.max(rho_label, dim=1)[1]

        
        if self.training and 'phi_label' in l.keys():
            phi_label = l['phi_label'].reshape(b).long()
            phi_noise = (torch.rand(b)>0.5).long()*torch.randint(-3,3,(b,)).long()
            phi_label = phi_label + phi_noise.to(phi_label.device)
            phi_label = torch.clamp(phi_label, 0, n-1)
        else:
            phi_label = torch.sigmoid(phi_prob)
            phi_label = torch.max(phi_label, dim=1)[1]


        rho_label = rho_label +0.5
        phi_label = phi_label +0.5

        init_rho = rho_label.reshape(b).float() * (2*np.pi/float(n))
        init_phi = phi_label.reshape(b).float() * (np.pi/float(n))

        zero = torch.zeros(b,1,1).to(init_rho.device)
        one = torch.ones(b,1,1).to(init_rho.device)

        init_rho = init_rho.reshape(b,1,1)
        m1 = torch.cat([
            torch.cat([torch.cos(init_rho), -torch.sin(init_rho), zero],dim=2),
            torch.cat([torch.sin(init_rho), torch.cos(init_rho), zero],dim=2),
            torch.cat([zero, zero, one],dim=2),
        ],dim=1)

        init_phi = init_phi.reshape(b,1,1)
        m2 = torch.cat([
            torch.cat([torch.cos(init_phi), zero, torch.sin(init_phi)],dim=2),
            torch.cat([zero, one, zero],dim=2),
            torch.cat([-torch.sin(init_phi), zero, torch.cos(init_phi)],dim=2),
        ],dim=1)

        return m1@m2






class I_Branch(nn.Module):
    def __init__(self, in_dim=256, ncls=1, resolution=32):
        super(I_Branch, self).__init__()
        self.ncls = ncls
        self.res = resolution

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, self.res//8, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6*self.ncls),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, vp_rot, cls=None):
        emb = self._get_transformed_feat(x, vp_rot)
        
        emb = self.conv(emb).squeeze(3).squeeze(2)
        r6d = self.mlp(emb)

        if self.ncls > 1:
            b = r6d.size(0)
            index = cls.reshape(b,1,1).expand(b,6,1)
            r6d = r6d.reshape(b,6,self.ncls)
            r6d = torch.gather(r6d, 2, index).squeeze(2)
        r = Ortho6d2Mat(r6d[:,0:3], r6d[:,3:6])
        return r

    def _get_transformed_feat(self, x, vp_rot):
        b,c,n,_ = x.size()
        assert n == self.res

        grid = torch.arange(n).float().to(x.device) + 0.5
        grid_rho = grid * (2*np.pi/float(n))
        grid_rho = grid_rho.reshape(1,n).repeat(n,1)
        grid_phi = grid * (np.pi/float(n))
        grid_phi = grid_phi.reshape(n,1).repeat(1,n)

        sph_xyz = torch.stack([
            grid_rho.cos() * grid_phi.sin(),
            grid_rho.sin() * grid_phi.sin(),
            grid_phi.cos(),
        ])

        sph_xyz = sph_xyz.reshape(1,3,-1).repeat(b,1,1)
        new_sph_xyz = vp_rot.transpose(1,2) @ sph_xyz

        sph_xyz = sph_xyz.transpose(1,2).contiguous().detach()
        new_sph_xyz = new_sph_xyz.transpose(1,2).contiguous().detach()
        x = x.reshape(b,c,n*n).contiguous()

        dist, idx = three_nn(sph_xyz, new_sph_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        new_x = three_interpolate(
            x, idx.detach(), weight.detach()
        ).reshape(b,c,n,n)
        return new_x




class I_Branch_Pair(nn.Module):
    def __init__(self, in_dim=256, ncls=1, resolution=32):
        super(I_Branch_Pair, self).__init__()
        self.ncls = ncls
        self.res = resolution

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, self.res//8, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 6*self.ncls),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, ref, vp_rot, cls=None):
        emb = self._get_transformed_feat(x, vp_rot)
        
        assert emb.shape == ref.shape
        emb = torch.cat([emb, ref], axis = 1)
        emb = self.conv(emb).squeeze(3).squeeze(2)
        r6d = self.mlp(emb)

        if self.ncls > 1:
            b = r6d.size(0)
            index = cls.reshape(b,1,1).expand(b,6,1)
            r6d = r6d.reshape(b,6,self.ncls)
            r6d = torch.gather(r6d, 2, index).squeeze(2)
        r = Ortho6d2Mat(r6d[:,0:3], r6d[:,3:6])
        return r

    def _get_transformed_feat(self, x, vp_rot):
        b,c,n,_ = x.size()
        assert n == self.res

        grid = torch.arange(n).float().to(x.device) + 0.5
        grid_rho = grid * (2*np.pi/float(n))
        grid_rho = grid_rho.reshape(1,n).repeat(n,1)
        grid_phi = grid * (np.pi/float(n))
        grid_phi = grid_phi.reshape(n,1).repeat(1,n)

        sph_xyz = torch.stack([
            grid_rho.cos() * grid_phi.sin(),
            grid_rho.sin() * grid_phi.sin(),
            grid_phi.cos(),
        ])

        sph_xyz = sph_xyz.reshape(1,3,-1).repeat(b,1,1)
        new_sph_xyz = vp_rot.transpose(1,2) @ sph_xyz

        sph_xyz = sph_xyz.transpose(1,2).contiguous().detach()
        new_sph_xyz = new_sph_xyz.transpose(1,2).contiguous().detach()
        x = x.reshape(b,c,n*n).contiguous()

        dist, idx = three_nn(sph_xyz, new_sph_xyz)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        new_x = three_interpolate(
            x, idx.detach(), weight.detach()
        ).reshape(b,c,n,n)
        return new_x

