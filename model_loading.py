import sys
from torchvision import models as torchvision_models
import torch
import torch.nn as nn
import timm
import transformers
import utils

def build_model(args):
    if args.pretrain_method == 'dino':
        # if "vit" in args.arch:
        #     model = transformers.dino_vit.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        #     print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
        # elif "xcit" in args.arch:
        #     model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        # elif args.arch in torchvision_models.__dict__.keys():
        #     model = torchvision_models.__dict__[args.arch](num_classes=0)
        #     model.fc = nn.Identity()
        # else:
        #     print(f"Architecture {args.arch} non supported")
        #     sys.exit(1)
        # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        model = timm.models.vision_transformer.vit_base_patch16_224_dino(pretrained=True)
        model.head = nn.Identity()

    # elif args.pretrain_method == 'mae':
    #     model = transformers.mae_vit.vit_base_patch16(global_pool=True)
    #     checkpoint = torch.load('./transformers/mae_finetuned_vit_base.pth', map_location='cpu')
    #     checkpoint_model = checkpoint['model']
    #     state_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #     # interpolate position embedding
    #     transformers.mae_vit.interpolate_pos_embed(model, checkpoint_model)
    #     _ = model.load_state_dict(checkpoint_model, strict=False)
    #     model.head = nn.Identity()

    elif args.pretrain_method == 'mae':
        model = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
        model.head = nn.Identity()

    elif args.pretrain_method == 'deit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        model.head = nn.Identity()

    elif args.pretrain_method == '1k':
        model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
        model.head = nn.Identity()
    
    elif args.pretrain_method == '21k':
        model = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True)
        model.head = nn.Identity()

    else:
        raise NotImplementedError
    return model