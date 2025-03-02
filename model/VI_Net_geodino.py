import torch
import torch.nn as nn
import torch.nn.functional as F
import math, open3d
import numpy as np
from module import SphericalFPN3, SphericalFPN4, V_Branch, I_Branch, I_Branch_Pair
from extractor_dino import ViTExtractor
from loss import SigmoidFocalLoss
from module import PointNet2MSG, KPConvFPN
from lib.sphericalmap_utils.smap_utils import Feat2Smap
from torchvision import transforms
from utils.rotation_utils import angle_of_rotation, Ortho6d2Mat
from utils.data_utils import global_fusion

    
def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis

class Net(nn.Module):
    def __init__(self, cfg, resolution=64, ds_rate=2, num_patches = 15):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution//ds_rate
        extractor = ViTExtractor('dinov2_vits14', 14, device = 'cuda')
        self.extractor =  torch.hub.load('/root/autodl-tmp/sgpose/hub/dinov2', 'dinov2_vits14',trust_repo=True, source='local').cuda()
    
        self.extractor_preprocess = transforms.Normalize(mean=extractor.mean, std=extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.pn2msg = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]], dim_in = 384+3)
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]], dim_in = 3)
        self.num_patches = num_patches
        
        self.kpcnet = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )
        
        self.global_encoding = global_fusion(128)

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.soft_max = nn.Softmax(dim=-1)

        # data processing
        self.feat2smap = Feat2Smap(self.res)
        self.feat2smap_drift = Feat2Smap(self.res//self.ds_rate)
 
        self.spherical_fpn = SphericalFPN3(ds_rate=self.ds_rate, dim_in1=1  , dim_in2=3 + 384, dim_in3 = 128)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim = 256)
        self.i_branch = I_Branch(resolution=self.ds_res, in_dim = 256)
        self.match_threshould = nn.Parameter(torch.tensor(-1.0, requires_grad=True))
        
    def extract_feature(self, rgb_raw):
        #import pdb;pdb.set_trace()
        rgb_raw = rgb_raw.permute(0,3,1,2)
        
        rgb_raw = self.extractor_preprocess(rgb_raw)
        with torch.no_grad():
            dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:,1:]
        dino_feature = dino_feature.reshape(dino_feature.shape[0],self.num_patches,self.num_patches,-1)
        
        return dino_feature.contiguous() 

    def inference(self,inputs):
        #import pdb;pdb.set_trace()
        pts= inputs['pts']
        rgb = inputs['rgb']
        b,rgb_h,rgb_w,_ = inputs['rgb_raw'].shape
        
        # superpoint
        points_f = []
        feats_c = []
        feats_f = []
        for i in range(len(inputs['lengths'])):
            an_points_c = inputs['pcd_points'][i][-1].detach()
            an_points_f = inputs['pcd_points'][i][0].detach()         
            pcd_features = inputs['pcd_feats'][i].detach()
            feats_list = self.kpcnet(pcd_features, inputs, i)     # [s1,s2,s3,s4]
            an_feats_c = feats_list[-1]     # superpoint's features
            an_feats_f = feats_list[0].unsqueeze(0)   
            points_f.append(an_points_f.unsqueeze(0))
            feats_c.append(an_feats_c)
            feats_f.append(an_feats_f)

        feats_f = torch.cat(feats_f, dim=0)
        points_f = torch.cat(points_f, dim=0)
        
        coords1 = points_f.permute(0, 2, 1)
        feats_s1 = feats_f.permute(0, 2, 1)
        feats_s1 = self.global_encoding(coords1, feats_s1, 20) # B, 128, N
        feats_s1 = feats_s1.permute(0, 2, 1)            # B, N, 128
        coords1 = coords1.permute(0, 2, 1)
        
        # dino
        rgb_raw = inputs['rgb_raw']
        feature = self.extract_feature(rgb_raw).reshape(b,240**2,-1)
        match_num = 100
        choose = inputs['choose'][:,:match_num]
        feature = feature[torch.arange(b)[:,None], 
                          choose.reshape(b,match_num),:]
        pts_raw = inputs['pts_raw']
        ptsf = pts_raw.reshape(b,(self.num_patches)**2,-1)[torch.arange(b)[:,None], choose,:]

        # mapping
        dis_map, rgb_map= self.feat2smap(pts, rgb)
        _, ref_map = self.feat2smap(ptsf, feature)
        #_, geo_map = self.feat2smap(points_f, feats_f)
        _, global_map = self.feat2smap(coords1, feats_s1)
        
        # SphericalFPN
        #x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map],dim = 1) , geo_map)
        x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map],dim = 1) , global_map)

        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob,{})
        
        ip_rot = self.i_branch(x, vp_rot)

        outputs = {
            'pred_rotation': vp_rot@ip_rot,
        }
        return outputs

    def forward(self, inputs):
        
        #import pdb;pdb.set_trace()
        #print(inputs)
 
        rgb=inputs['rgb']
        pts= inputs['pts']
        b,rgb_h,rgb_w,_ = inputs['rgb_raw'].shape
        
        # surperpoint
        points_f = []
        feats_c = []
        feats_f = []
        for i in range(len(inputs['batch_size'])):
            an_points_c = inputs['pcd_points'][i][-1].detach()
            an_points_f = inputs['pcd_points'][i][0].detach()         
            pcd_features = inputs['pcd_feats'][i].detach()
            feats_list = self.kpcnet(pcd_features, inputs, i)     # [s1,s2,s3,s4]
            an_feats_c = feats_list[-1]    # superpoint's features
            an_feats_f = feats_list[0].unsqueeze(0)  
            #print(an_feats_f.shape)
            points_f.append(an_points_f.unsqueeze(0))
            feats_c.append(an_feats_c)
            feats_f.append(an_feats_f)
        
        #import pdb;pdb.set_trace()
        feats_f = torch.cat(feats_f, dim=0)
        points_f = torch.cat(points_f, dim=0)
        #print(feats_f.shape)
        #print(points_f.shape)
        
        coords1 = points_f.permute(0, 2, 1)
        feats_s1 = feats_f.permute(0, 2, 1)
        feats_s1 = self.global_encoding(coords1, feats_s1, 20) # B, 128, N
        feats_s1 = feats_s1.permute(0, 2, 1)            # B, N, 128
        coords1 = coords1.permute(0, 2, 1)
        
        # dino
        rgb_raw = inputs['rgb_raw']
        feature = self.extract_feature(rgb_raw).reshape(b,240**2,-1)
        match_num = 100
        choose = inputs['choose'][:,:match_num]
        pts_raw = inputs['pts_raw']
        pts_raw = pts_raw.reshape(b,(self.num_patches)**2,-1)[torch.arange(b)[:,None], choose,:]
        rgb_raw = rgb_raw.reshape(b,(self.num_patches)**2,-1)[torch.arange(b)[:,None], choose,:]
        ptsf = pts_raw
        feature = feature[torch.arange(b)[:,None], choose,:]

        # mapping
        #import pdb;pdb.set_trace()           
        dis_map, rgb_map= self.feat2smap(pts, rgb)
        _, ref_map = self.feat2smap(ptsf, feature)
        #_, geo_map = self.feat2smap(points_f, feats_f)
        _, global_map = self.feat2smap(coords1, feats_s1)

        # spherical fpn
        #x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map],dim = 1) , geo_map)
        x = self.spherical_fpn(dis_map, torch.cat([rgb_map, ref_map],dim = 1) , global_map)
        
        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob,{})

        ip_rot = self.i_branch(x, vp_rot)
        
        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'pred_vp_rotation': pred_vp_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob,
           
        }
        return outputs


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.l1loss = nn.L1Loss()
        self.smoothl1loss = SmoothL1Dis
        self.sfloss = SigmoidFocalLoss()

    def forward(self, pred, gt):
        rho_prob = pred['rho_prob']
        rho_label = F.one_hot(gt['rho_label'].squeeze(1), num_classes=rho_prob.size(1)).float()
        rho_loss = self.sfloss(rho_prob, rho_label).mean()
        pred_rho = torch.max(torch.sigmoid(rho_prob),1)[1]
        rho_acc = (pred_rho.long() == gt['rho_label'].squeeze(1).long()).float().mean() * 100.0

        phi_prob =  pred['phi_prob']
        phi_label = F.one_hot(gt['phi_label'].squeeze(1), num_classes=phi_prob.size(1)).float()
        phi_loss = self.sfloss(phi_prob, phi_label).mean()
        pred_phi = torch.max(torch.sigmoid(phi_prob),1)[1]
        phi_acc = (pred_phi.long() == gt['phi_label'].squeeze(1).long()).float().mean() * 100.0
        
        vp_loss = rho_loss + phi_loss
        ip_loss = self.l1loss(pred['pred_rotation'], gt['rotation_label'])

        residual_angle = angle_of_rotation(pred['pred_rotation'].transpose(1,2) @ gt['rotation_label'])
        
        loss = self.cfg.vp_weight * vp_loss + ip_loss
        return {
            'loss': loss,
            'vp_loss': vp_loss,
            'ip_loss': ip_loss,
            'rho_acc': rho_acc,
            'phi_acc': phi_acc,
            'residual_angle_loss':residual_angle.mean(),
            
            '5d_loss':  (residual_angle<5).sum()/residual_angle.shape[0],
            

        }
