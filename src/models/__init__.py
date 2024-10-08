from models.segmentor.zegclip import ZegCLIP
from models.segmentor.Proposed_segmentor import ProposedCLIPSegmentor

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg

from models.losses.atm_loss import SegLossPlus

from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
