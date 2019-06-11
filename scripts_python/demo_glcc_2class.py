# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

result_show = False
result_save = True

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
#from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.net_utils_saba import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.alexnet import alexnet
from model.faster_rcnn.resnet18 import Resnet18
from model.faster_rcnn.inceptionv3 import Inceptionv3
from model.faster_rcnn.densenet121 import Dense121

#from model.faster_rcnn.glcc_2class import GLCC
from model.faster_rcnn.glcc_2class_accuracy88 import GLCC

import pdb

import matplotlib.pyplot as plt
import time

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

    
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='saba_20171219_train', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, alexnet, res18, inceptionv3, dense121',
                      default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="/home/usrs/nagayosi/Program_sababox/faster-rcnn.pytorch/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default='/home/usrs/nagayosi/Program_sababox/Dataset/saba_20180119/Test/Lined/Images')
                     #default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_false')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=False, type=bool)
                      #action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--show', dest='show',
                      help='show result',
                      action='store_true')
  parser.add_argument('--epoch', dest='epoch', help='model s epoch when testing', default=100, type=int)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

if __name__ == '__main__':

  args = parse_args()
  result_show = args.show
  
  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  #load_name = os.path.join(input_dir,
  #  'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  
  #load_name = os.path.join(input_dir, 'vgg16_caffe.pth')

  #pascal_classes = np.asarray([
  #    '__background__',
  #    'Blue mackerel', 'Chub mackerel', 'Hybrid',
  #    'Blue mackerel redline', 'Chub mackerel redline', 'Hybrid redline'])
  pascal_classes = np.asarray(['__background__', 'mackerel', 'redline'])
  saba_classes = np.asarray(['__background__', 'Blue mackerel', 'Chub mackerel', 'Hybrid'])

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)

  elif args.net == 'res18':
    fasterRCNN = Resnet18(pascal_classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
    
  elif args.net == 'alexnet':
    fasterRCNN = alexnet(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    
  elif args.net == 'inceptionv3':
    fasterRCNN = Inceptionv3(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
     
  elif args.net == 'dense121':
    fasterRCNN = Dense121(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
      
  else:
    print("network is not defined")
    pdb.set_trace()

  frcn_load_name = 'models_2class/{}_glcc/saba_20171219_train/glcc_faster_rcnn_1_{}_833.pth'.format(args.net, args.epoch)
  glcc_load_name = 'models_2class/{}_glcc/saba_20171219_train/glcc_1_{}_833.pth'.format(args.net, args.epoch)
  result_dir = 'result_{}_glcc_2class'.format(args.net)

  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
  fasterRCNN.create_architecture()
  fasterRCNN.load_state_dict(torch.load(frcn_load_name)['model'])

  glcc = GLCC(fasterRCNN, use_cuda=True, classes=saba_classes, net=args.net, pretrained=False, class_agnostic=args.class_agnostic)
  glcc.load_state_dict(torch.load(glcc_load_name)['model'])

  """
  tar = torch.load(load_name)
  state_dict = tar['model']
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  unpred_keys = ['FRCN', 'RPN', 'se.','p.', 's_']
  for k, v in state_dict.items():
      print(k)
      if ('FRCN' not in k) and ('RCNN' not in k):
          new_state_dict[k] = v
          print(k)
  glcc.load_state_dict(new_state_dict)
  """
  #params = {}
  #for k, v in torch.load(load_name)['model'].items():
  #    if 'FRCN' not in k:
  #        params[k] = v
  #glcc.load_state_dict(params)
  
  print("frcn load checkpoint >> {}".format(frcn_load_name))

  if args.cuda > 0:
    checkpoint = torch.load(frcn_load_name)
  else:
    checkpoint = torch.load(frcn_load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (frcn_load_name))
  print("load checkpoint %s" % (glcc_load_name))  

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()
    glcc.cuda()

  fasterRCNN.eval()
  glcc.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

  webcam_num = args.webcam_num
  # Set up webcam or get image directories
  if webcam_num >= 0 :
    cap = cv2.VideoCapture(webcam_num)
    num_images = 0
  else:
    from glob import glob
    #imglist = os.listdir(args.image_dir+'/*.jpg')
    imglist = glob(args.image_dir + '/*.jpg')
    num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  P_table = np.zeros((4, 4), dtype=np.int)
  P_table_l = np.zeros((4, 4), dtype=np.int)
  N_det = 0.
  N_det_l = 0.
  N_all = num_images
  sum_time = 0.


  while (num_images > 0):
      total_tic = time.time()
      if webcam_num == -1:
        num_images -= 1

      # Get image from the webcam
      if webcam_num >= 0:
        if not cap.isOpened():
          raise RuntimeError("Webcam could not open. Please check connection.")
        ret, frame = cap.read()
        im_in = np.array(frame)
      # Load the demo image
      else:
        #im_file = os.path.join(args.image_dir, imglist[num_images])
        im_file = imglist[num_images]
        # im = cv2.imread(im_file)
        im_in = np.array(imread(im_file))
      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]


      # read ground truth
      gt_path = im_file.replace('Images', 'Annotations')[:-4] + '.txt'
      with open(gt_path, 'r') as f:
          for l in f.readlines():
              l = l.strip()
              _cls, x1, y1, x2, y2 = list(map(int, l.split(',')))
              if 'goma' in im_file:
                  cls = 1
              elif 'masaba' in im_file:
                  cls = 2
              elif 'hybrid' in im_file:
                  cls = 3
                  
              if _cls < 5:
                  gt = np.array((x1, y1, x2, y2, cls))
              elif _cls >= 5:
                  gt_l = np.array((x1, y1, x2, y2, cls+3))

              
      t1 = time.time()

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
      im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
      gt_boxes.data.resize_(1, 1, 5).zero_()
      num_boxes.data.resize_(1).zero_()

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, glcc_prob, glcc_loss, glcc_gt = glcc(im_data, im_info, gt_boxes, num_boxes)
      
      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()

      all_dets = np.ndarray((0, 5), dtype=np.float32)
      all_dets_cls = []
      all_dets_cls_ind = []

      # left-top-x left-top-y right-bottom-x right-bottom-y, score, cls
      region_g = np.ndarray((0,6), dtype=np.float32)
      region_l = np.ndarray((0,6), dtype=np.float32)

      t2 = time.time()
      sum_time += (t2 - t1)

      if vis:
          im2show = np.copy(im)
      plt.close()
      plt.figure(num=os.path.basename(imglist[num_images]))
      plt.imshow(im2show[..., ::-1])
      ax = plt.axes()
          
      t1 = time.time()

      # get global region
      inds = torch.nonzero(scores[:,1] > thresh).view(-1)
      if inds.numel() > 0:
        cls_scores = scores[:,1][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if args.class_agnostic:
          cls_boxes = pred_boxes[inds]
        else:
          cls_boxes = pred_boxes[inds][:, 4:8]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1).cpu().numpy()
        det_g = np.r_[cls_dets[np.argmax(cls_dets[..., -1])], [1]]
        region_g = np.vstack((region_g, det_g))
      
      
      # get local region
      inds = torch.nonzero(scores[:,2]>thresh).view(-1)
      if inds.numel() > 0:
        cls_scores = scores[:,2][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if args.class_agnostic:
          cls_boxes = pred_boxes[inds]
        else:
          cls_boxes = pred_boxes[inds][:, 8:12]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        det_l = np.r_[cls_dets[np.argmax(cls_dets[...,-1])], [2]]
        region_l = np.vstack((region_l, det_l))

      if len(region_g) > 0:
        region_g = region_g[np.argmax(region_g[..., -2])]
        pred_ind = glcc_prob.data.cpu().numpy().argmax()

      t2 = time.time()
      sum_time += (t2 - t1)

      # choice max probability global region
      if len(region_g) > 0:
        det = region_g
        #pred_ind = glcc_prob.data.cpu().numpy().argmax()
        # evaluation between detection and gt 
        iou_x1 = np.maximum(det[0], gt[0])
        iou_y1 = np.maximum(det[1], gt[1])
        iou_x2 = np.minimum(det[2], gt[2])
        iou_y2 = np.minimum(det[3], gt[3])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area= iou_w * iou_h
        dets_saba_area = np.maximum((det[2] - det[0]) * (det[3] - det[1]), 0)
        gt_area = np.maximum((gt[2] - gt[0]) * (gt[3] - gt[1]), 0)
        iou = iou_area / (dets_saba_area + gt_area - iou_area)

        if iou >= 0.5:
            N_det += 1.
            P_table[gt[-1], int(pred_ind)] += 1.
            if int(pred_ind) == int(gt[-1]):
                color = 'red'
                #P_table[gt[-1], int(pred_ind)] += 1.
            else:
                color = 'blue'
                #P_table[gt[-1], 0] += 1.
        else:
            color = 'blue'

            P_table[gt[-1], 0] += 1.
        if vis:
            _d = tuple(int(np.round(x)) for x in det[:4])
            score = det[-2]
            ax.add_patch(
                plt.Rectangle((_d[0], _d[1]), _d[2]-_d[0], _d[3]-_d[1],
                              fill=False, edgecolor=color, linewidth=1.))
            ax.text(_d[0], _d[1] - 2,
                    '{:s} {:.3f}'.format(saba_classes[int(pred_ind)], score),
                    bbox=dict(facecolor=color, alpha=0.5),
                    fontsize=14, color='white')
            #cv2.rectangle(im2show, _d[:2], _d[2:4], color, 2)
            #cv2.putText(im2show, "{} : {:.3f}".format(pascal_classes[int(pred_ind)], score), (_d[0], _d[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), thickness=2)
      else:
          P_table[gt[-1], 0] += 1.
            
      # Get Local region
      if len(region_l) > 0:
        region_l = region_l[np.argmax(region_l[..., -2])]
        det = region_l
        # evaluation between detection and gt 
        iou_x1 = np.maximum(det[0], gt_l[0])
        iou_y1 = np.maximum(det[1], gt_l[1])
        iou_x2 = np.minimum(det[2], gt_l[2])
        iou_y2 = np.minimum(det[3], gt_l[3])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area= iou_w * iou_h
        dets_l_area = np.maximum((det[2] - det[0]) * (det[3] - det[1]), 0)
        gt_area = np.maximum((gt_l[2] - gt_l[0]) * (gt_l[3] - gt_l[1]), 0)
        iou = iou_area / (dets_l_area + gt_area - iou_area)

        if iou >= 0.5:
            N_det_l += 1.
            if int(det[-1]) == int(gt_l[-1]):
                color = 'red'
                P_table_l[gt_l[-1]-3, int(det[-1])-3] += 1.
            else:
                color = 'blue'
                P_table_l[gt_l[-1]-3, 0] += 1.
        else:
            color = 'blue'
            P_table_l[gt_l[-1]-3, 0] += 1
      
        if vis:
            _d = tuple(int(np.round(x)) for x in det[:4])
            score = det[-2]
            ax.add_patch(
                plt.Rectangle((_d[0], _d[1]), _d[2]-_d[0], _d[3]-_d[1],
                              fill=False, edgecolor=color, linewidth=1.))
            ax.text(_d[0], _d[3] - 2,
                    '{:s} {:.3f}'.format(pascal_classes[int(det[-1])], score),
                    bbox=dict(facecolor=color, alpha=0.5),
                    fontsize=14, color='white')
            #cv2.rectangle(im2show, _d[:2], _d[2:4], color, 2)
            #cv2.putText(im2show, "{} : {:.3f}".format(pascal_classes[int(det[-1])], score), (_d[0], _d[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,255), thickness=2)
      else:
          P_table_l[gt_l[-1]-3, 0] += 1.
                
      #if vis:
      #  im2show = vis_detections(im2show, all_dets_cls, all_dets, thresh=0.8)

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          #result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
          result_path = os.path.join(result_dir, imglist[num_images].split('/')[-1][:-4] + '.jpg')
          if result_save:
              plt.savefig(result_path)
              #cv2.imwrite(result_path, im2show)
          if result_show:
              #plt.figure(num=os.path.basename(imglist[num_images]))
              #plt.imshow(im2show[..., ::-1])
              plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
              ax.axis('off')
              ax.tick_params(labelbottom=False, bottom=False)
              ax.tick_params(labelleft=False, left=False)
              plt.show()
              #cv2.imshow(os.path.basename(imglist[num_images]), im2show)
              #cv2.waitKey(0)
              #cv2.destroyAllWindows()
      else:
          im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          cv2.imshow("frame", im2showRGB)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()

  print()
  # compute precision global
  accu = 0
  for i in range(1, 4):
    accu += P_table[i,i]

  print('   bg, BM, CM, HY')
  print(P_table)
  print("Global:Precision >> {} ({} / {})".format(accu / N_all, accu, N_all))
  print("Global:Detection >> {} ({} / {})".format(N_det / N_all, N_det, N_all))

  # compute precition local
  accu = 0
  for i in range(1, 4):
    accu += P_table_l[i,i]

  print('   bg, BM, CM, HY')
  print(P_table_l)
  print("Local:Precision >> {} ({} / {})".format(accu / N_all, accu, N_all))
  print("Local:Detection >> {} ({} / {})".format(N_det_l / N_all, N_det_l, N_all))

  print("average processing time >> {} ({} sec/ {}image)".format(sum_time / N_all, sum_time, N_all))
