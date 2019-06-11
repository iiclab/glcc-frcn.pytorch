from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN
#from model.faster_rcnn.resnet import resnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np
import matplotlib.pyplot as plt

from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.net_utils import _affine_grid_gen

class GLCC(nn.Module):
    def __init__(self, FRCN, use_cuda, classes, net='res101', pretrained=False, class_agnostic=False):
        #resnet.__init__(self, classes=classes, num_layers=num_layers, pretrained=pretrained, class_agnostic=class_agnostic)
        super(GLCC, self).__init__()
        self.FRCN = FRCN
        if net == 'res101':
            self.fc_num = 2048
            self.roipool = 7
        elif net == 'res18':
            self.fc_num = 512
            self.roipool = 7
        elif net == 'alexnet':
            self.fc_num = 512
            self.roipool = 6
        elif net == 'vgg16':
            self.fc_num = 1024
            self.roipool = 7
        self.glcc_conv1 = nn.Conv2d(self.fc_num, 512, (1,1))
        self.glcc_fc1 = nn.Linear(self.roipool * self.roipool * 512, 1024)
        self.glcc_fc2 = nn.Linear(1024, 1024)
        self.glcc_fc_out = nn.Linear(1024, 4)
        self.use_cuda = use_cuda
        self.class_agnostic = class_agnostic
        self.classes = classes
  
    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)
        
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label \
            = self.FRCN(im_data, im_info, gt_boxes, num_boxes)

        # get global and local region from Faster R-CNN

        base_feat = self.FRCN.RCNN_base(im_data)

        #print(rois.data.cpu().numpy())
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        box_deltas = self.FRCN._bbox_pred.data

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            if self.class_agnostic:
                if self.use_cuda > 0:
                    box_deltas = box_deltas.view(-1,4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1,4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) * torch.FlaotTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if self.use_cuda > 0:
                    box_deltas = box_deltas.view(-1,4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1,4) * torhc.FlaotTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        # get global region
        thresh = 0.00
        
        region_g = np.ndarray((0, 5))
        region_l = np.ndarray((0, 5))
        for j in range(1, 4):
            inds = torch.nonzero(scores[:,j]>=thresh).view(-1)
            inds_l = torch.nonzero(scores[:,j+3]>=thresh).view(-1)
            #print(inds)
            if inds.numel() > 0 and inds_l.numel() > 0:
                cls_scores = scores[:,j][inds]
                cls_scores_l = scores[:,j+3][inds_l]
                #print(cls_scores)
                #print(cls_scores_l)
                _, order = torch.sort(cls_scores, 0, True)
                _, order_l = torch.sort(cls_scores_l, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds]
                    cls_boxes_l = pred_boxes[inds_l]
                else:
                    cls_boxes = pred_boxes[inds][:,j*4:(j+1)*4]
                    cls_boxes_l = pred_boxes[inds_l][:,(j+3)*4:(j+4)*4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets_l = torch.cat((cls_boxes_l, cls_scores_l.unsqueeze(1)), 1)

                cls_dets = cls_dets[order]
                cls_dets_l = cls_dets_l[order_l]

                region_g = np.vstack((region_g, cls_dets[np.argmax(cls_dets[..., -1])]))
                region_l = np.vstack((region_l, cls_dets_l[np.argmax(cls_dets_l[..., -1])]))

        #print(cls_dets)
        #print(pred_boxes)

        # if true, then show detection global and local region
        if False:
            print(region_g)
            print(region_l)
            im = im_data.cpu().numpy()[0]
            im = np.transpose(im, (1,2,0))[..., ::-1]
            im -= im.min()
            im /= im.max()
            plt.imshow(im.astype(np.float))
            ax = plt.axes()
            ax.add_patch(
                plt.Rectangle((region_g[0, 0], region_g[0, 1]),
                              region_g[0, 2] - region_g[0, 0],
                              region_g[0, 3] - region_g[0, 1], fill=False,
                              edgecolor='red', linewidth=1))

            ax.add_patch(
                plt.Rectangle((region_l[0, 0], region_l[0, 1]),
                              region_l[0, 2] - region_l[0, 0],
                              region_l[0, 3] - region_l[0, 1], fill=False,
                              edgecolor='yellow', linewidth=1))
            plt.show()

        rois_g = np.zeros((1,1,5), dtype=np.float32)
        rois_g[0,0,1:5] = region_g[0, :4] / 16.
        rois_l = np.zeros((1,1,5), dtype=np.float32)
        rois_l[0,0,1:5] = region_l[0, :4] / 16.

        GPU = 0
        rois_g = torch.tensor(rois_g, dtype=torch.float).to(GPU)
        rois_l = torch.tensor(rois_l, dtype=torch.float).to(GPU)
        
        # global region
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois_g.view(-1, 5), base_feat.size()[2:], self.FRCN.grid_size)
            grid_yx = torch.stack([grid_xy.data[...,1], grid_xy.data[...,0]], 3).contiguous()
            pooled_feat_g = self.FRCN.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_g = F.max_pool2d(pooled_feat_g, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_g = self.FRCN.RCNN_roi_align(base_feat, rois_g.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_g = self.FRCN.RCNN_roi_pool(base_feat, rois_g.view(-1, 5))

        # local region
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois_l.view(-1, 5), base_feat.size()[2:], self.FRCN.grid_size)
            grid_yx = torch.stack([grid_xy.data[..., 1], grid_xy.data[...,0]], 3).contiguous()
            pooled_feat_l = self.FRCN.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_l = F.max_pool2d(pooled_feat_l, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_l = self.FRCN.RCNN_roi_align(base_feat, rois_l.view(-1,5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_l = self.FRCN.RCNN_roi_pool(base_feat, rois_l.view(-1,5))
            
        #print(pooled_feat_g.cpu().detach().numpy().shape)
        x = torch.cat((pooled_feat_g, pooled_feat_l), dim=1)
        #print(x.cpu().detach().numpy().shape)
        x = self.glcc_conv1(x)
        x = F.relu(x)
        x = x.view(-1, self.roipool * self.roipool * 512)
        x = self.glcc_fc1(x)
        s = F.relu(x)
        x = nn.Dropout2d()(x)
        x = self.glcc_fc2(x)
        x = F.relu(x)
        x = nn.Dropout2d()(x)
        x = self.glcc_fc_out(x)
        
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, x

    

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.FRCN.RCNN_base.eval()
            self.FRCN.RCNN_base[5].train()
            self.FRCN.RCNN_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.FRCN.RCNN_base.apply(set_bn_eval)
            self.FRCN.RCNN_top.apply(set_bn_eval)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            if truncated:
                m.weight.data.normal_().fmod_(2).mnul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.glcc_conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.glcc_fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.glcc_fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.glcc_fc_out, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
