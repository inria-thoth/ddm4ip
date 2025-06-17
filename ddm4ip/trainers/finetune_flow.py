import os
import warnings
import pickle

from omegaconf import DictConfig
import torch
import torch.utils.data

from ddm4ip.data.base import Batch, Datasplit, init_dataset
from ddm4ip.losses.flow_matching import FlowMatchingLoss
from ddm4ip.networks.unets import init_unet_with_defaults
from ddm4ip.trainers.base import BaseTrainer, init_trts_dataloaders
from ddm4ip.trainers.plots import plot_imggen
from ddm4ip.utils import distributed
from ddm4ip.utils.easy_dict import print_module_summary
from ddm4ip.utils.phema import PowerFunctionEMA


class FinetuneFlowTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.plot_batch = None
        self.flow_nn_ema = None

    def init_datasets(self, cfg: DictConfig):
        return {
            Datasplit.TRAIN: init_dataset(cfg, split=Datasplit.TRAIN, is_paired=False),
            Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True)
        }

    def init_dataloaders(self, cfg, dsets):
        return init_trts_dataloaders(
            cfg, dsets, start_idx=self.start_global_step, is_infinite=self.is_training
        )

    def init_models(self, cfg, dsets, device): # type: ignore
        dataset = dsets[Datasplit.TRAIN]
        assert dataset is not None
        flow_cfg = cfg["models"]["flow"]
        flow_nn = init_unet_with_defaults(
            flow_cfg,
            img_size=dataset.corrupt_img_size,
            cond_ch=dataset.corrupt_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        if flow_cfg.get("pretrained_path") is not None:
            if not os.path.isfile(flow_cfg["pretrained_path"]):
                raise FileNotFoundError(
                    f"Pretrained flow model path '{flow_cfg['pretrained_path']}' is not a file."
                )
            # import sys
            # sys.path.append(f'{os.path.expanduser("~")}/diffusion')  # TODO: This is crap
            if flow_cfg["pretrained_path"].endswith(".pt"):
                prtr_ckpt = torch.load(
                    flow_cfg["pretrained_path"], weights_only=False, map_location="cpu"
                )
                flow_nn.load_state_dict(prtr_ckpt["ema"]["emas"][-1], strict=True)
                del prtr_ckpt
            elif flow_cfg["pretrained_path"].endswith(".pkl"):
                with open(flow_cfg["pretrained_path"], "rb") as fh:
                    in_pkl_data = pickle.load(fh)
                    prtr_ckpt = in_pkl_data['ema'].to(torch.float32)
                    flow_nn.load_state_dict(prtr_ckpt.state_dict(), strict=False)
                    del prtr_ckpt
            else:
                raise RuntimeError(
                    f"Unrecognized file extension for '{flow_cfg['pretrained_path']}'"
                )
        else:
            warnings.warn("finetune_flow was run without 'pretrained_path' config. "
                          "From-scratch training will be done instead!")
        # # NOTE: This is a stupid hack. Needed to support loading base models
        # #       with different resolution than the `corrupt_img_size`
        # flow_nn.img_resolution = dataset.corrupt_img_size
        flow_nn = flow_nn.to(device)
        self.flow_nn_ema = None
        if cfg.get("ema") is not None:
            # Sync EMA between ranks so that EMA is properly initialized.
            # Needed because module parameters are automatically synced by DDP after this point only.
            model_sync(flow_nn)
            self.flow_nn_ema = PowerFunctionEMA(flow_nn, stds=cfg["ema"]["stds"])

        if distributed.get_rank() == 0:
            print("---------------------------FLOW NN------------------------------")
            cond_ch = dataset.corrupt_conditioning_channels
            print_module_summary(flow_nn, [
                torch.zeros([self.batch_size, *dataset.corrupt_img_size], device=device),
                torch.ones([self.batch_size], device=device),
                torch.zeros([self.batch_size, dataset.label_dim], device=device),
                torch.zeros([self.batch_size, cond_ch, *dataset.corrupt_img_size[1:]], device=device)
            ], max_nesting=2)
        return {
            "flow_nn": flow_nn
        }

    def init_loss(self, cfg, models):
        if cfg['loss']['name'] == 'flow_matching':
            # flow match on the y (corrupted images)
            loss = FlowMatchingLoss(cfg, clean_images=False, **models)
        else:
            raise ValueError(f"Loss of type '{cfg['loss']['name']}' invalid.")
        return loss

    def validate_batch(self, val_batch):
        return self.loss_optim.val_loss(self, val_batch)

    def on_train_step_finished(self):
        super().on_train_step_finished()
        # Update EMA
        if self.flow_nn_ema is not None:
            self.flow_nn_ema.update(cur_nimg=self.global_step, batch_size=self.global_batch_size)

    def get_extra_state(self):
        if self.flow_nn_ema is not None:
            return {
                "ema": self.flow_nn_ema.state_dict()
            }
        return {}

    def set_extra_state(self, state):
        if "ema" in state:
            if self.flow_nn_ema is None:
                warnings.warn("Checkpoint state contains 'ema' key but current model is not setup for EMA.")
            else:
                self.flow_nn_ema.load_state_dict(state["ema"])
        elif self.flow_nn_ema is not None:
            raise RuntimeError("Current model expects EMA data but checkpoint does not contain any.")

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_batch is None:
            # This ensures we always use the same batch for plotting.
            if (ds := dsets.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                ds = dsets[Datasplit.TRAIN]
                assert ds is not None
            # Don't use the dataloader to get around data-sampling artefacts.
            self.plot_batch = Batch.collate_fn([ds[i] for i in range(self.batch_size)])
        flow_nn = self.flow_nn_ema.get()[-1][0] if self.flow_nn_ema is not None else models["flow_nn"]

        dset = dsets[Datasplit.TEST]
        assert dset is not None
        p1 = plot_imggen(cfg, flow_nn, self.plot_batch, dset.corrupt_img_size, dset, self.device)
        if p1 is not None and distributed.get_rank() == 0:
            self.log_image(p1, 'diffusion/generated', tb_writer)


@torch.no_grad()
def model_sync(model):
    if torch.distributed.is_initialized():
        psums = []
        for name, param in model.named_parameters():
            psums.append(param.mean())
            torch.distributed.broadcast(param, src=0)
