import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils import kaiming_init
from mmdet.models.builder import HEADS

from .bbox_head import BBoxHead
from ...utils.transformer import FeatEmbed


@HEADS.register_module()
class KEHead(BBoxHead):
    def __init__(self,
                 num_classes=294,
                 num_bbox_classes=47,
                 in_dims=256,
                 bbox_cls_fc_dims=1024,
                 patch_size=1,
                 know_matrix_path=None,
                 know_update=False,
                 cross_encoder=None,
                 self_decoder=None,
                 cross_decoder=None,
                 *arg, **kwargs):
        super(KEHead, self).__init__(*arg, **kwargs)

        self.num_classes = num_classes
        self.num_bbox_classes = num_bbox_classes
        self.in_dims = in_dims
        self.half_dims = in_dims // 2

        ''' Visual context encoding'''
        # bbox embedding
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(self.half_dims, self.half_dims, kernel_size=3, padding=1),
            nn.GroupNorm(4, self.half_dims),
            nn.ReLU())
        # gt_bbox embedding
        self.human_bbox_conv = nn.Sequential(
            nn.Conv2d(self.in_dims, self.half_dims, kernel_size=1),
            nn.GroupNorm(4, self.half_dims),
            nn.ReLU())
        # encoder
        self.encoder = build_transformer_layer_sequence(cross_encoder)
        # bbox embed
        self.bbox_embed = FeatEmbed(self.roi_feat_size, patch_size)

        '''Geometry context encoding'''
        # bbox loc embed
        self.loc_embed = nn.Linear(4, in_dims)

        '''Explicit knowledge'''
        # knowleage matrix
        knowledge_matrix = np.load(know_matrix_path)
        human_knowledge = np.zeros((1, 294), dtype=np.float32)
        knowledge_matrix_all = np.concatenate([knowledge_matrix, human_knowledge], axis=0)
        if know_update is False:
            self.knowledge_matrix = nn.Parameter(
                torch.tensor(knowledge_matrix_all, dtype=torch.float32), requires_grad=False)
        else:
            self.knowledge_matrix = nn.Parameter(
                torch.tensor(knowledge_matrix_all, dtype=torch.float32), requires_grad=True)

        '''Part identifier'''
        # cls embed
        self.cls_embed = nn.Sequential(
            nn.Linear(bbox_cls_fc_dims, in_dims),
            nn.ReLU(),
            nn.Linear(in_dims, in_dims))
        
        '''Conditional projection'''
        # fc_attr_cls_
        self.fc_attr_cls_weight = nn.Parameter(torch.randn(self.num_classes, in_dims))
        kaiming_init(self.fc_attr_cls_weight)
        # ffn
        self.fc_attr_transform = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm((in_dims)),
            nn.ReLU(),
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm((in_dims)))
        # decoder
        self.self_decoder = build_transformer_layer_sequence(self_decoder)
        self.cross_decoder = build_transformer_layer_sequence(cross_decoder)

        # for no used paramter
        self.fc_cls = nn.Identity()

    def forward(self,
                bbox_feats,
                human_bbox_feats,
                pos_gt_labels,
                pos_cls_scores,
                bbox_locs,
                fc_cls_weight):
        
        ''' Visual context encoding'''
        # bbox conv
        b, c, h, w = bbox_feats.shape
        bbox_src_feats = bbox_feats[:, :c//2, :, :]
        bbox_src_feats = self.bbox_conv(bbox_src_feats)
        human_bbox_feats = self.human_bbox_conv(human_bbox_feats)
        # qkv
        query = bbox_feats[:, c//2:, :, :].flatten(2).permute(2, 0, 1)
        key = value = human_bbox_feats.flatten(2).permute(2, 0, 1)
        # for bbox feat
        bbox_rela_feats = self.encoder(query=query,
                                       key=key,
                                       value=value)
        bbox_rela_feats = bbox_rela_feats.permute(1, 2, 0).reshape(b, self.half_dims, h, w)
        bbox_feats = torch.cat([bbox_src_feats, bbox_rela_feats], dim=1)
        bbox_feats = self.bbox_embed(bbox_feats).permute(1, 0, 2)

        '''Geometry context encoding'''
        # for bbox location
        bbox_locs = self.loc_embed(bbox_locs).unsqueeze(0)

        '''Categorical attribute representation'''
        # for bbox cls 
        with torch.no_grad():
            bbox_cls_feats = pos_cls_scores.unsqueeze(-1) * fc_cls_weight.unsqueeze(0)
        knowledge_matrix = self.knowledge_matrix[
            :self.num_bbox_classes-1, :].unsqueeze(0).repeat(b, 1, 1)
        bbox_cls_feats = torch.matmul(
            knowledge_matrix.permute(0, 2, 1), bbox_cls_feats[:, :self.num_bbox_classes-1, :])
        bbox_cls_feats = self.cls_embed(bbox_cls_feats).permute(1, 0, 2)

        '''Attribute queris'''
        # decoder query
        attr_cls_weight = self.fc_attr_transform(self.fc_attr_cls_weight)
        attr_knowledge = self.knowledge_matrix[pos_gt_labels]
        attr_knowledge = (attr_knowledge > 0).float()
        decoder_query = attr_knowledge.unsqueeze(-1) * attr_cls_weight.unsqueeze(0)
        decoder_query = decoder_query.permute(1, 0, 2)

        '''Conditional projection'''
        # for input x
        x = torch.cat([bbox_feats, bbox_cls_feats, bbox_locs, decoder_query], dim=0)
        # transformer
        memory = self.self_decoder(
            query=x,
            key=None,
            value=None)
        cross_query_embed = memory[-self.num_classes:, :, :]

        out_dec = self.cross_decoder(
            query=cross_query_embed,
            key=memory[:-self.num_classes, :, :],
            value=memory[:-self.num_classes, :, :])
        out_dec = out_dec.permute(1, 0, 2)
        cls_score = self.fc_attr_cls_weight.unsqueeze(0) * out_dec

        return cls_score.sum(dim=-1)
