import os
import warnings

from omegaconf import DictConfig
import torch

from ddm4ip.data.base import Batch, Datasplit, init_dataset
from ddm4ip.losses.di_y import DiffInstructOnY
from ddm4ip.networks import init_kernel_net
from ddm4ip.networks.unets import init_unet_with_defaults
from ddm4ip.trainers.base import BaseTrainer, init_trts_dataloaders
from ddm4ip.trainers.plots import plot_conditioning_hist, plot_explicit_kernels, plot_imggen, plot_implicit_kernels, plot_spacevarying_kernels
from ddm4ip.utils import distributed
from ddm4ip.utils.easy_dict import print_module_summary


class DiffinstructOpTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.plot_batch = None
        self.corrupt_img_size: tuple[int, int, int] | None = None

    def init_datasets(self, cfg: DictConfig):
        if self.is_training:
            return {
                Datasplit.TRAIN: init_dataset(cfg, split=Datasplit.TRAIN, is_paired=False),
                Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True)
            }
        return {
            Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True)
        }

    def init_dataloaders(self, cfg, dsets):
        # No infinite data-loader during evaluation
        return init_trts_dataloaders(
            cfg, dsets, start_idx=self.start_global_step, is_infinite=self.is_training
        )

    def init_models(self, cfg, dsets, device):
        dataset = self.get_any_dset(dsets)
        assert dataset is not None
        # Load the pre-trained model
        prt_flow_cfg = cfg["models"]["pretrained_flow"]
        if "path" not in prt_flow_cfg:
            raise ValueError(
                "'pretrained_flow' configuration must contain "
                "'path' key pointing to the pretrained model checkpoint."
            )
        if not os.path.isfile(prt_path := prt_flow_cfg["path"]):
            raise FileNotFoundError(
                f"Pretrained flow model path '{prt_path}' is not a file."
            )
        ckpt = torch.load(prt_path, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            state = ckpt["ema"]["emas"][-1]
            print(f"Loading pretrained flow model. EMA state with std: {ckpt['ema']['stds'][-1]}")
        elif "flow_nn" in ckpt:
            # This is the output of `finetune_flow`
            state = ckpt["flow_nn"]["state_dict"]
        else:
            raise ValueError(list(ckpt.keys()))

        # Initialize the kernel NN
        kernel_nn = init_kernel_net(
            cfg["models"]["kernel"],
            img_size=dataset.clean_img_size,
            cond_ch=dataset.clean_conditioning_channels,
            label_dim=dataset.label_dim,
            is_output_noisy=True,
        ).to(device)
        kernel_nn_fake_inputs = [
            torch.zeros([self.batch_size, *dataset.clean_img_size], device=device),
            torch.zeros([self.batch_size], device=device),
            torch.zeros([self.batch_size, dataset.label_dim], device=device),
            torch.zeros([self.batch_size, dataset.clean_conditioning_channels, *dataset.clean_img_size[1:]], device=device)
        ]
        kernel_nn_fake_output = kernel_nn(*kernel_nn_fake_inputs)
        pred_corrupt_size = (kernel_nn_fake_output.shape[1], kernel_nn_fake_output.shape[2], kernel_nn_fake_output.shape[3])
        self.corrupt_img_size = pred_corrupt_size

        prtr_flow_nn = init_unet_with_defaults(
            cfg=prt_flow_cfg,
            img_size=pred_corrupt_size,
            cond_ch=dataset.corrupt_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        prtr_flow_nn.load_state_dict(state)
        prtr_flow_nn = prtr_flow_nn.to(device)
        # Create the auxiliary model to match the pre-trained model
        init_from_pretrained = False
        if (aux_flow_cfg := cfg["models"].get("aux_flow")) is None:
            aux_flow_cfg = prt_flow_cfg
            init_from_pretrained = True
        aux_flow_nn = init_unet_with_defaults(
            aux_flow_cfg,
            img_size=pred_corrupt_size,
            cond_ch=dataset.corrupt_conditioning_channels,
            label_dim=dataset.label_dim,
        )
        if init_from_pretrained:
            aux_flow_nn.load_state_dict(state)
        aux_flow_nn = aux_flow_nn.to(device)

        if distributed.get_rank() == 0:
            print("-----------------------AUXILIARY FLOW NN------------------------")
            cond_ch = dataset.corrupt_conditioning_channels
            print_module_summary(aux_flow_nn, [
                torch.zeros([self.batch_size, *pred_corrupt_size], device=device),
                torch.ones([self.batch_size], device=device),
                torch.zeros([self.batch_size, dataset.label_dim], device=device),
                torch.zeros([self.batch_size, cond_ch, *pred_corrupt_size[1:]], device=device)
            ], max_nesting=2)
            print("-------------------------KERNEL FLOW NN-------------------------")
            print_module_summary(kernel_nn, kernel_nn_fake_inputs, max_nesting=2)
        return {
            "aux_flow_nn": aux_flow_nn,
            "prtr_flow_nn": prtr_flow_nn,
            "kernel_nn": kernel_nn,
        }

    def init_loss(self, cfg, models):
        loss = DiffInstructOnY(cfg, **models)
        return loss

    def validate_batch(self, val_batch):
        return self.loss_optim.val_loss(self, val_batch)

    def get_any_dset(self, dsets):
        return next(d for d in dsets.values() if d is not None)

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_batch is None:
            # This ensures we always use the same batch for plotting.
            if (ds := dsets.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                ds = dsets[Datasplit.TRAIN]
                assert ds is not None
            # Don't use the dataloader to get around data-sampling artefacts.
            # messing around with indexes to get decent images for plotting
            self.plot_batch = Batch.collate_fn([ds[i] for i in range(self.batch_size)])

        dset = self.get_any_dset(dsets)
        assert self.corrupt_img_size is not None
        p1 = plot_imggen(cfg, models["aux_flow_nn"], self.plot_batch, self.corrupt_img_size, dset, self.device)
        if p1 is not None and distributed.get_rank() == 0:
            self.log_image(p1, 'diffusion/generated', tb_writer)

        p2, p2_ker = None, None
        try:
            p2, p2_ker = plot_spacevarying_kernels(
                dset.corruption, dset.corrupt_img_size[-1], self.models["kernel_nn"], self.batch_size, self.device
            )
        except TypeError:
            pass
        if p2 is None:
            p2, p2_ker = plot_explicit_kernels(dset, models["kernel_nn"], self.plot_batch, self.batch_size, self.device)
        if p2 is not None and distributed.get_rank() == 0:
            self.log_image(p2, 'eval/explicit-kernels', tb_writer)
        if p2_ker is not None and distributed.get_rank() == 0:
            tb_writer.add_tensor('eval/kernel', p2_ker, self.global_step)
        p3 = plot_implicit_kernels(self.plot_batch, models["kernel_nn"])
        if p3 is not None and distributed.get_rank() == 0:
            self.log_image(p3, 'eval/implicit-kernels', tb_writer)
        p4 = plot_conditioning_hist(self.loss_optim)
        if p4 is not None and distributed.get_rank() == 0:
            self.log_image(p4, 'debug/conditioning-distribution', tb_writer)

        if hasattr(models["kernel_nn"], "sigma"):
            tb_writer.add_scalar('eval/learned_noise', models["kernel_nn"].sigma.item(), self.global_step)
