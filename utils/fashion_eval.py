from fashionpedia.fp import Fashionpedia
from fashionpedia.fp_eval import FPEval

# annotation and prediction file names here
anno_file = "data/fashionpedia/instances_attributes_val2020.json"
res_file = "work_dirs/ke_rcnn_r50_fpn_fashion_1x/ke_rcnn_r50_fpn_fashion_1x_val_result.attr.json"

# initialize Fashionpedia groudtruth and prediction api
fpGt=Fashionpedia(anno_file)
fpDt=fpGt.loadRes(res_file)
imgIds=sorted(fpGt.getImgIds())

# run evaluation
fp_eval = FPEval(fpGt, fpDt, 'bbox')
fp_eval.params.imgIds  = imgIds
imgIds=sorted(fpGt.getImgIds())

fp_eval.run()
fp_eval.print() # print out result using both Iou AND F1 constraint
# fp_eval.print(f1=False) # print out result using f1 only
# fp_eval.print(iou=False) # print out result using f1 only
