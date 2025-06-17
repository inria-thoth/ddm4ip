from typing import Any, Iterable, Mapping

import deepinv
import torch
import torchvision.transforms.v2.functional as TF

from ddm4ip.data.base import Batch
from ddm4ip.degradations.blur import Blur
from ddm4ip.degradations.downsampling import Downsampling
from ddm4ip.degradations.varpsf import PerPatchInterpolatedBlur, PerPixelBlur
from ddm4ip.losses.base import AbstractLoss
from ddm4ip.psf.psf import norm_sum_to_one
from ddm4ip.utils.metrics import LPIPS, calc_psnr, calc_ssim
from ddm4ip.utils.torch_utils import img2patches, patches2img


class DeepInvLoss(AbstractLoss):
    def __init__(self,
                 config: Mapping[str, Any],
                 physics: deepinv.physics.Physics,
                 deepinv_model: torch.nn.Module,
                 kernel_nn: torch.nn.Module):
        super().__init__(has_val_loss=True)
        self.deepinv_model = deepinv_model
        self.kernel_nn = kernel_nn
        self.physics = physics
        self.lpips = LPIPS(net_type="alex").cuda()
        loss_cfg = config["loss"]
        self.crop_filters: int = loss_cfg.get("crop_filters", 0)
        self.patch_size: int | None = loss_cfg.get("patch_size", None)
        self.padding: int = loss_cfg.get("padding", 0)
        self.patch_batch_size: int = loss_cfg.get("patch_batch_size", 1)

    def run_model(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch.corrupt is not None

        data_resolution = batch.corrupt.shape[-2], batch.corrupt.shape[-1]
        if self.patch_size is not None and (self.patch_size != data_resolution[0] or self.patch_size != data_resolution[1]):
            # Input is in full resolution,
            # we need to convert batch data to patches
            if batch.corrupt.shape[0] > 1:
                raise ValueError(
                    f"DeepInv loss does not support batch-sizes > 1 "
                    f"when full-images are provided. Found batch size of {batch.corrupt.shape[0]}"
                )
            image_c, image_h, image_w = batch.corrupt.shape[-3:]
            if batch.corrupt_conditioning is not None:
                img_and_cond = torch.cat([batch.corrupt, batch.corrupt_conditioning], dim=-3)
            else:
                img_and_cond = batch.corrupt
            patches = img2patches(img_and_cond, patch_size=self.patch_size + self.padding, stride=self.patch_size)
            restored_patches, corruption_filters = self.run_on_iterable(patches, image_c)
            restored_patches = [p.squeeze(0) for p in restored_patches]
            patch_size_mult = restored_patches[0].shape[-1] / patches[0].shape[-1]
            def _crop(x):
                return TF.center_crop(x, output_size=[x.shape[-2] - int(self.padding * patch_size_mult), x.shape[-1] - int(self.padding * patch_size_mult)])
            restored_image = patches2img(
                restored_patches,
                stride=int(self.patch_size * patch_size_mult),
                imgh=int(image_h * patch_size_mult),
                imgw=int(image_w * patch_size_mult),
                func=_crop
            )
            restored_image = restored_image.unsqueeze(0)  # add back the batch dimension
        else:
            restored_image, corruption_filters = self.run_on_data(
                y=batch.corrupt.cuda(),
                x=batch.clean.cuda() if batch.clean is not None else None,
                conditioning=batch.corrupt_conditioning.cuda() if batch.corrupt_conditioning is not None else None
            )
            restored_image = restored_image.cpu()
            corruption_filters = corruption_filters.cpu()
        # Update physics based on conditioning.
        return restored_image, corruption_filters

    def run_on_iterable(self, y_and_cond: Iterable[torch.Tensor], y_channels: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        out_patches: list[torch.Tensor] = []
        out_filters = []
        sub_batch = []
        for patch in y_and_cond:
            sub_batch.append((patch[..., :y_channels, :, :], patch[..., y_channels:, :, :]))
            if len(sub_batch) >= self.patch_batch_size:
                y_stack = torch.cat([b[0] for b in sub_batch], dim=0).cuda()
                cond_stack = torch.cat([b[1] for b in sub_batch], dim=0).cuda()
                if cond_stack.numel() == 0:
                    cond_stack = None
                restored, filters = self.run_on_data(y_stack, x=None, conditioning=cond_stack)
                out_patches.extend(restored.cpu().split(1, dim=0))
                out_filters.append(filters.cpu())
                sub_batch = []
        if len(sub_batch) > 0:
            y_stack = torch.cat([b[0] for b in sub_batch], dim=0).cuda()
            cond_stack = torch.cat([b[1] for b in sub_batch], dim=0).cuda()
            if cond_stack.numel() == 0:
                cond_stack = None
            restored, filters = self.run_on_data(y_stack, x=None, conditioning=cond_stack)
            out_patches.extend(restored.cpu().split(1, dim=0))
            out_filters.append(filters.cpu())

        # out_patches = torch.cat(out_patches, 0)  # N, C, pS, pS
        out_filters = torch.cat(out_filters, 0)
        return out_patches, out_filters

    def run_on_data(self, y: torch.Tensor, x: torch.Tensor | None, conditioning: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        filters = self.kernel_nn.get_kernel(img=y, conditioning=conditioning)
        if self.crop_filters > 0:
            filters = filters[..., self.crop_filters:-self.crop_filters, self.crop_filters:-self.crop_filters]
            filters = norm_sum_to_one(filters)
        if isinstance(self.physics, (Blur, PerPatchInterpolatedBlur, Downsampling)):
            self.physics.update_parameters(filter=filters)
        elif isinstance(self.physics, PerPixelBlur):
            self.physics.update_parameters(filters=filters)
        else:
            raise NotImplementedError(f"Physics of type {type(self.physics)} not implemented in DeepInvLoss.")
        x_net = self.deepinv_model(
            y, self.physics, x_gt=x, compute_metrics=False
        )
        x_net = torch.clamp(x_net, 0, 1)
        return x_net, filters

    def compute_img_metrics(self, x_recon: torch.Tensor, x_gt: torch.Tensor):
        metrics = {
            "lpips": self.lpips(x_gt.cuda(), x_recon.cuda()),
            "psnr": calc_psnr(x_gt, x_recon),
            "ssim": calc_ssim(x_gt, x_recon),
        }
        return metrics

    def __call__(self, trainer, batch: Batch):
        pred, filters = self.run_model(batch)
        loss_info = {}
        if batch.clean is not None:
            metrics = self.compute_img_metrics(pred, batch.clean)
            loss_info.update({f"loss/{k}": v for k, v in metrics.items()})
        return loss_info

    @torch.no_grad()
    def val_loss(self, trainer, batch: Batch):
        loss_dict = self(trainer, batch)
        return {
            k.replace("loss", "val_loss"): v for k, v in loss_dict.items()
        }

    @torch.no_grad()
    def val_loss_with_output(self, trainer, batch: Batch) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        pred, filters = self.run_model(batch)
        loss_info = {}
        if batch.clean is not None:
            metrics = self.compute_img_metrics(pred, batch.clean)
            loss_info.update({f"val_loss/{k}": v for k, v in metrics.items()})
        return loss_info, pred, filters

    def is_full_step_complete(self) -> bool:
        return True

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
