import mmcv
import numpy as np
from .builder import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class FashionPedia(CocoDataset):

    CLASSES = ('shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan',
               'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
               'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
               'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings',
               'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
               'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 
               'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon',
               'rivet', 'ruffle', 'sequin', 'tassel')
    num_attribute = 294

    def __init__(self, *arg, with_human=False, **kwargs):
        super(FashionPedia, self).__init__(*arg, **kwargs)
        self.with_human = with_human

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_attributes = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info): # iterative by human
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

            # 因为标注的时候从234直接跳到281了，为了方便实现，这里做一个减法, 同时少了284这个 id
            attribute_ids = np.array(ann['attribute_ids'], dtype=np.int64)
            attribute_ids = np.where(attribute_ids < 281, attribute_ids, attribute_ids-46)
            attribute_ids = np.where(attribute_ids < 239, attribute_ids, attribute_ids-1)

            attributes = np.zeros((1, self.num_attribute), dtype=np.int64)
            attributes[:, attribute_ids] = 1
            gt_attributes.append(attributes)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_attributes = np.array(gt_attributes, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_attributes = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        
        if self.with_human:
            min_x = gt_bboxes[:, 0].min()
            max_x = gt_bboxes[:, 2].max()
            min_y = gt_bboxes[:, 1].min()
            max_y = gt_bboxes[:, 3].max()
            
            if max_x > min_x and max_y > min_y:
                human_bbox = np.array([[min_x-10, min_y-10, max_x+10, max_y+10]], dtype=np.float32)
                human_bbox[:, 0::2] = np.clip(human_bbox[:, 0::2], 0, img_info['width'])
                human_bbox[:, 1::2] = np.clip(human_bbox[:, 1::2], 0, img_info['height'])
                human_label = np.array([len(self.CLASSES)-1], dtype=np.int64)
                human_attributes = np.zeros((1, 1, self.num_attribute), dtype=np.int64)
                gt_bboxes = np.concatenate((gt_bboxes, human_bbox), axis=0)
                gt_labels = np.concatenate((gt_labels, human_label), axis=0)
                gt_attributes = np.concatenate((gt_attributes, human_attributes), axis=0)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            attributes=gt_attributes,
            masks=gt_masks_ann)

        return ann

    def results2json(self, results, outfile_prefix):
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        elif isinstance(results[0], dict):
            json_results = self._attribute_2json(results)
            result_files['attr'] = f'{outfile_prefix}.attr.json'
            mmcv.dump(json_results, result_files['attr'])
        else:
            raise TypeError('invalid type of results')
        return result_files
    
    def _attribute_2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det_result, segm_result, attribute_result = results[idx].keys()
            det_result = results[idx][det_result]
            segm_result = results[idx][segm_result]
            attribute_result = results[idx][attribute_result]
            if self.with_human:
                det_result = det_result[:len(self.CLASSES)-1]
                segm_result = segm_result[:len(self.CLASSES) - 1]
                attribute_result = attribute_result[:len(self.CLASSES)-1]
            for label in range(len(det_result)):
                bboxes = det_result[label]
                segms = segm_result[label]
                attributes = attribute_result[label]
                for i in range(bboxes.shape[0]):
                    attribute_ids = attributes[i]
                    attribute_ids = \
                        np.where(attribute_ids <= 234, attribute_ids, attribute_ids+46)
                    attribute_ids = \
                        np.where(attribute_ids <= 283, attribute_ids, attribute_ids+1)
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['attribute_ids'] = attribute_ids.tolist()
                    data['score'] = float(bboxes[i][4])
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results
