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
from model.nms.nms_wrapper import nms

class GLCC(nn.Module):
    def __init__(self, FRCN, use_cuda, classes, net='res101', pretrained=False, class_agnostic=False):
        #resnet.__init__(self, classes=classes, num_layers=num_layers, pretrained=pretrained, class_agnostic=class_agnostic)
        super(GLCC, self).__init__()
        self.FRCN = FRCN
        self.FRCN.eval()
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
        feat = 4096
        self.glcc_conv1 = nn.Conv2d(self.fc_num, 512, (1,1))
        self.glcc_fc1 = nn.Linear(self.roipool * self.roipool * 512, feat)
        self.glcc_fc2 = nn.Linear(feat, feat)
        self.glcc_fc_out = nn.Linear(feat, 4)
        self.use_cuda = use_cuda
        self.class_agnostic = class_agnostic
        self.classes = classes

        # training hyper parameters
        self.minibatch = 1
        self.gt_iou = 0.4
        self.nms_iou = 0.3

        self.rois_g = torch.FloatTensor(1)
        self.rois_l = torch.FloatTensor(1)
        self.glcc_gt = torch.LongTensor(1)

        if use_cuda:
            self.rois_g = self.rois_g.cuda()
            self.rois_l = self.rois_l.cuda()
            self.glcc_gt = self.glcc_gt.cuda()

        self.rois_g = torch.autograd.Variable(self.rois_g)
        self.rois_l = torch.autograd.Variable(self.rois_l)
        self.glcc_gt = torch.autograd.Variable(self.glcc_gt)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, opt_frcn=None):
        batch_size = im_data.size(0)
        
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        #self.FRCN.eval()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
            RCNN_loss_cls, RCNN_loss_bbox, rois_label \
            = self.FRCN(im_data, im_info, gt_boxes, num_boxes)

        if opt_frcn is not None:
            opt.zero_grad()

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
                cls_dets_l = cls_dets_l[order]

                region_g = np.vstack((region_g, cls_dets))
                region_l = np.vstack((region_l, cls_dets_l))
                """
                keep = nms(cls_dets, 0.9, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                keep = nms(cls_dets_l, 0.9, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets_l = cls_dets_l[keep.view(-1).long()]

                cls_dets = cls_dets[order]
                cls_dets_l = cls_dets_l[order_l]

                sort_ind = np.argsort(cls_dets[...,-1])
                high_ind = sort_ind[-self.minibatch//2:]
                low_ind = sort_ind[:self.minibatch//2]
                region_g = np.vstack((region_g. cls_dets[high_ind]))
                region_g = np.vstack((region_g, cls_dets[low_ind]))]

                sort_ind = np.argsort(cls_dets_l[..., -1])
                high_ind = sort_ind[-self.minibatch//2:]
                low_ind = sort_ind[:self.minibatch//2]
                region_l = np.vstack((region_l, cls_dets_l[high_ind]))
                reigon_l = np.vstack((region_l, cls_dets_l[low_ind]))
                """
                #region_g = np.vstack((region_g, cls_dets[np.argmax(cls_dets[..., -1])]))
                #region_l = np.vstack((region_l, cls_dets_l[np.argmax(cls_dets_l[..., -1])]))

        if not self.training:
            self.minibatch = 1
                
        if self.training:
            if self.minibatch % 2 == 0:
                high_ind = self.minibatch // 2
                low_ind = self.minibatch // 2
            elif self.minibatch == 1:
                high_ind = 1
                low_ind = 0
            else:
                high_ind = self.minibatch // 2 + 1
                low_ind = self.minibatch // 2

            keep = nms(torch.tensor(region_g).cuda(), self.nms_iou, force_cpu=not cfg.USE_GPU_NMS)
            if type(keep) is not list:
                keep = keep.view(-1).long()
            region_g = region_g[keep]
            sort_ind = np.argsort(region_g[..., -1])
            high_ind_g = sort_ind[-high_ind:]
            low_ind_g = sort_ind[:low_ind]
            
            keep = nms(torch.tensor(region_l).cuda(), self.nms_iou, force_cpu=not cfg.USE_GPU_NMS)
            if type(keep) is not list:
                keep = keep.view(-1).long()
            region_l = region_l[keep]
            sort_ind = np.argsort(region_l[..., -1])
            high_ind_l = sort_ind[-high_ind:]
            low_ind_l = sort_ind[:low_ind]
            
            high_num = min(len(high_ind_g), len(high_ind_l))
            high_ind_g = high_ind_g[:high_num]
            high_ind_l = high_ind_l[:high_num]

            low_num = min(len(low_ind_g), len(low_ind_l))
            low_ind_g = low_ind_g[:low_num]
            low_ind_l = low_ind_l[:low_num]

            proposal_g = np.vstack((region_g[high_ind_g], region_g[low_ind_g]))
            proposal_l = np.vstack((region_l[high_ind_l], region_l[low_ind_l]))

            #self.proposal_g.data.resize_(proposal_g.size()).copy_(proposal_g)
            #self.proposal_l.data.resize_(proposal_l.size()).copy_(proposal_l)

            gt_boxes = gt_boxes.cpu().numpy()[0, :2]

            gt_g = gt_boxes[np.where(gt_boxes[...,-1] < 4)[0]]
            gt_l = gt_boxes[np.where(gt_boxes[...,-1] >= 4)[0]]

            # compute pare ground truth
            def compute_iou(ps, gt, th=0.5):
                iou_x1 = np.maximum(ps[..., 0], gt[0])
                iou_y1 = np.maximum(ps[..., 1], gt[1])
                iou_x2 = np.minimum(ps[..., 2], gt[2])
                iou_y2 = np.minimum(ps[..., 3], gt[3])
                iou_w = np.maximum(iou_x2 - iou_x1, 0)
                iou_h = np.maximum(iou_y2 - iou_y1, 0)
                iou_area = iou_w * iou_h
                gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                p_area = (ps[..., 2] - ps[..., 0]) * (ps[..., 3] - ps[..., 1])
                overlap = iou_area / (gt_area + p_area - iou_area)
                count = np.zeros((ps.shape[0]), dtype=int)
                count[overlap >= self.gt_iou] += 1
                return count
            
            cou = compute_iou(proposal_g, gt_g[0]) + compute_iou(proposal_l, gt_l[0])

            ## 2019.2.13
            glcc_gt = np.zeros((proposal_g.shape[0]), dtype=int)
            glcc_gt[cou==2] = gt_g[0,-1]
            #glcc_gt[:] = gt_g[0, -1]
            glcc_gt = torch.tensor(glcc_gt, dtype=torch.long).cuda()
            self.glcc_gt.data.resize_(glcc_gt.size()).copy_(glcc_gt)

        else:
            # test phase
            proposal_g = region_g[np.argmax(region_g[..., -1])][None, ...]
            proposal_l = region_l[np.argmax(region_l[..., -1])][None, ...]
            #self.proposal_g.data.resize_(proposal_g.size()).copy_(proposal_g.size())
            #self.proposal_l.data.resize_(proposal_l.size()).copy_(proposal_l.size())
            
        # if true, then show detection global and local region
        if False:
            gt_boxes = gt_boxes.astype(np.int)
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

            ax.add_patch(
                plt.Rectangle((gt_boxes[0, 0], gt_boxes[0, 1]),
                              gt_boxes[0, 2] - gt_boxes[0, 0],
                              gt_boxes[0, 3] - gt_boxes[0, 1], fill=False,
                              edgecolor='green', linewidth=1))
            ax.add_patch(
                plt.Rectangle((gt_boxes[1, 0], gt_boxes[1, 1]),
                              gt_boxes[1, 2] - gt_boxes[1, 0],
                              gt_boxes[1, 3] - gt_boxes[1, 1], fill=False,
                              edgecolor='white', linewidth=1))
            plt.show()

        rois_g = np.zeros((1,proposal_g.shape[0],5), dtype=np.float32)
        rois_g[0,:,1:5] = proposal_g[:, :4]
        #rois_g /= 16.
        rois_l = np.zeros((1,proposal_l.shape[0],5), dtype=np.float32)
        rois_l[0,:,1:5] = proposal_l[:, :4]
        #rois_l /= 16.
        rois_g = torch.tensor(rois_g, dtype=torch.float).cuda()
        rois_l = torch.tensor(rois_l, dtype=torch.float).cuda()
        self.rois_g.data.resize_(rois_g.size()).copy_(rois_g)
        self.rois_l.data.resize_(rois_l.size()).copy_(rois_l)
        # global region
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(self.rois_g.view(-1, 5), base_feat.size()[2:], self.FRCN.grid_size)
            grid_yx = torch.stack([grid_xy.data[...,1], grid_xy.data[...,0]], 3).contiguous()
            pooled_feat_g = self.FRCN.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_g = F.max_pool2d(pooled_feat_g, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_g = self.FRCN.RCNN_roi_align(base_feat, self.rois_g.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_g = self.FRCN.RCNN_roi_pool(base_feat, self.rois_g.view(-1, 5))

        # local region
        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(self.rois_l.view(-1, 5), base_feat.size()[2:], self.FRCN.grid_size)
            grid_yx = torch.stack([grid_xy.data[..., 1], grid_xy.data[...,0]], 3).contiguous()
            pooled_feat_l = self.FRCN.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_l = F.max_pool2d(pooled_feat_l, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_l = self.FRCN.RCNN_roi_align(base_feat, self.rois_l.view(-1,5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_l = self.FRCN.RCNN_roi_pool(base_feat, self.rois_l.view(-1,5))
            
        #print(pooled_feat_g.cpu().detach().numpy().shape)
        x = torch.cat((pooled_feat_g, pooled_feat_l), dim=1)
        #print(x.cpu().detach().numpy().shape)
        x = self.glcc_conv1(x)
        x = F.relu(x)
        x = x.view(-1, self.roipool * self.roipool * 512)
        x = self.glcc_fc1(x)
        x = F.relu(x)
        x = nn.Dropout()(x)
        x = self.glcc_fc2(x)
        x = F.relu(x)
        x = nn.Dropout()(x)
        glcc_out = self.glcc_fc_out(x)

        if self.training:
            glcc_gt = torch.tensor(glcc_gt, dtype=torch.long).cuda()
            glcc_loss = F.cross_entropy(glcc_out, self.glcc_gt)
        else:
            glcc_loss = 0.
            glcc_gt = None
        
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, glcc_out, glcc_loss, glcc_gt

    

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
