# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import logging

import torch
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss,iBOTPatchLoss

# logger = logging.getLogger("dinov3")
dino_out_dim = 8192
dino_loss = DINOLoss(dino_out_dim)
ibot_patch_loss = iBOTPatchLoss(8192)
is_distillation_enabled = False


def debug_forward(
        model, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ):
        del ignored_kwargs
        metrics_dict = {}

        # Shapes
        n_global_crops = 2
        n_local_crops = 8  # self.cfg.crops.local_crops_number
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]
        metrics_dict["teacher_temp"] = teacher_temp #add teacher_temp to logging

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        # Teacher output (will trigger an all-gather to unshard)
        teacher_global = get_teacher_output(
            model,
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        # Student output (will trigger an all-gather to unshard)
        student_global, student_local = get_student_output(
            model,
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        # Return total weighted loss and a dict of metrics to log
        return teacher_global, student_global

@torch.no_grad()
def get_teacher_output(
    model,
    images,
    *,
    upperbound,
    mask_indices_list,
    teacher_temp,
    n_masked_patches_tensor,
):
    n_crops, B, rgb, H, W = images.shape
    images = images.flatten(0, 1)

    backbone_out = model.teacher.backbone(images, is_training=True)
    cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
    reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
    ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

    # IBOT head only on patches that are masked for the student
    buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
    masked_patch_after_head = model.teacher.ibot_head(buffer)

    # DINO head on CLS tokens
    cls_after_head = model.teacher.dino_head(cls)  # [n_crops * B, K]

    # Center with sinkhorn-knopp
    cls_centered = dino_loss.sinkhorn_knopp_teacher(
        cls_after_head, teacher_temp=teacher_temp
    )  # [n_crops * B, K]
    cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
    masked_patch_centered = ibot_patch_loss.sinkhorn_knopp_teacher(
        masked_patch_after_head,
        teacher_temp=teacher_temp,
        n_masked_patches_tensor=n_masked_patches_tensor,
    )  # [n_masked_patches, K]

    return {
        "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
        "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
        "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
        "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
        "cls_centered": cls_centered,  # [n_crops, B, K]
        "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
    }

def get_student_output(model, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
    n_global_crops, B, rgb, H, W = global_crops.shape
    n_local_crops, B, rgb, H, W = local_crops.shape

    global_crops = global_crops.flatten(0, 1)

    # Forward global and local crops through the student backbone jointly
    global_out, local_out = model.student.backbone(
        [global_crops, local_crops.flatten(0, 1)],
        masks=[masks if not is_distillation_enabled else None, None],
        is_training=True,
    )
    g_cls, g_reg, g_patch = (
        global_out["x_norm_clstoken"],
        global_out["x_storage_tokens"],
        global_out["x_norm_patchtokens"],
    )
    l_cls, l_reg, l_patch = (
        local_out["x_norm_clstoken"],
        local_out["x_storage_tokens"],
        local_out["x_norm_patchtokens"],
    )

    # IBOT head only on masked patches
    masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
    global_masked_patch_after_head = model.student.ibot_head(masked_patches_pre_head)

    # DINO head on CLS tokens (all in one pass)
    buffer = [
        g_cls,  # [n_global_crops * B, D]
        l_cls,  # [n_local_crops * B, D]
    ]
    sizes = [x.shape[0] for x in buffer]
    buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
    buffer = model.student.dino_head(buffer)  # [n_global_crops * B + n_local_crops * B, K]
    buffer = torch.split_with_sizes(buffer, sizes, dim=0)

    global_out = {
        "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
        "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
        "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
        "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
        "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
        "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
    }
    local_out = {
        "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
        "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
        "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
        "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
    }

    return global_out, local_out