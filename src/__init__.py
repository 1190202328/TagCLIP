from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.backbone.text_encoder import CLIPTextEncoder
from models.losses.atm_loss import SegLossPlus
from models.segmentor.Proposed_segmentor import ProposedCLIPSegmentor
from models.segmentor.zegclip import ZegCLIP
