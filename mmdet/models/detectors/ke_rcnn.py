# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class KERCNN(TwoStageDetector):
    """Implementation of `KE R-CNN` """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 attr_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(KERCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        if train_cfg is not None:
            if isinstance(train_cfg.rcnn, list):
                rcnn_train_cfg = train_cfg.rcnn[-1]
            else:
                rcnn_train_cfg = train_cfg.rcnn
        else:
            rcnn_train_cfg = None
        attr_head.update(train_cfg=rcnn_train_cfg)
        attr_head.update(test_cfg=test_cfg.rcnn)
        attr_head.pretrained = pretrained
        self.attr_head = build_head(attr_head)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # roi head
        roi_results = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs)
        losses.update(roi_results['roi_losses'])

        # attr head
        if isinstance(self.roi_head.bbox_head, nn.ModuleList):
            fc_cls_weight = self.roi_head.bbox_head[-1].fc_cls.weight
        else:
            fc_cls_weight = self.roi_head.bbox_head.fc_cls.weight

        attr_losses = self.attr_head.forward_train(
            x,
            img_metas,
            roi_results['sampling_results'],
            roi_results['bbox_results'],
            fc_cls_weight,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs)
        losses.update(attr_losses)

        return losses
    
    def simple_test(self, img, img_metas, rescale, **kwargs):

        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)

        # roi head forward
        results = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

        if isinstance(self.roi_head.bbox_head, nn.ModuleList):
            fc_cls_weight = self.roi_head.bbox_head[-1].fc_cls.weight
        else:
            fc_cls_weight = self.roi_head.bbox_head.fc_cls.weight

        # attribute forward
        attr_results = self.attr_head.simple_test(
            x, 
            results['garments_bboxes'],
            results['garments_scores'],
            results['garments_labels'],
            fc_cls_weight,
            img_metas,
            rescale=rescale)

        return [dict(det_results=results['det_results'], segm_results=results['segm_results'],
                    attr_results=attr_results)]
