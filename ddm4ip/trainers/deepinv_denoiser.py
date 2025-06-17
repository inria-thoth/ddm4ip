import copy
import math
import os
import pickle
import warnings

from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
import torch
from torchvision.utils import make_grid
import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.dpir import get_DPIR_params
from deepinv.utils.parameters import get_GSPnP_params

from ddm4ip.data.base import Batch, Datasplit, init_dataset
from ddm4ip.degradations.degradation import init_noise
from ddm4ip.losses.deepinv_loss import DeepInvLoss
from ddm4ip.networks.blur_nets import BlurDegradationType
from ddm4ip.trainers.base import BaseTrainer, init_trts_dataloaders
from ddm4ip.trainers.plots import plot_spacevarying_kernels
from ddm4ip.utils import distributed
from ddm4ip.utils.deepinv_utils import DinvReconstructorWrapper, GSPnP, WienerSolver, DPSWrapper
from ddm4ip.utils.diffpir import DiffPIR
from ddm4ip.utils.torch_utils import write_img_pt


class DeepinvDenoiserTrainer(BaseTrainer):
    """
    Inverse problem solver wrapping [DeepInv](https://deepinv.github.io/) implementations.
    This class takes in a **paired dataset**, which is generated through our learned
    inverse operator. It then uses DeepInv to attempt to invert this operator through standard
    methods:
     - GSPnP (copied from the [deepinv tutorial](https://deepinv.github.io/deepinv/auto_examples/plug-and-play/demo_RED_GSPnP_SR.html))

    Be careful when instantiating the dataset:
     - it should use the "true" corruption process to generate samples.
     - the modeled corruption process should be used to initialize the DeepInv model
    """
    def __init__(self):
        super().__init__()
        self.plot_dloader = None
        self.physics = None

    def init_datasets(self, cfg: DictConfig):
        return {
            # This dataset should be paired!
            Datasplit.TEST: init_dataset(cfg, split=Datasplit.TEST, is_paired=True)
        }

    def init_dataloaders(self, cfg, dsets):
        # No infinite data-loader because we want to loop through the test
        # dataset once only.
        return init_trts_dataloaders(
            cfg, dsets, start_idx=self.start_global_step, is_infinite=False
        )

    def init_gspnp(self, prior_model, noise_level):
        # Parameters here are mostly copied from the deepinv tutorial on GSPnP.
        prior = GSPnP(denoiser=prior_model)
        lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params("deblur", noise_level)
        params_algo = {
            "stepsize": stepsize,
            "g_param": sigma_denoiser,
            "lambda": lamb,
        }
        # we want to output the intermediate PGD update to finish with a denoising step.
        def custom_output(X):
            return X["est"][1]
        optim_params = {
            "iteration": "PGD",
            "g_first": True,
            "max_iter": max_iter,
            "params_algo": params_algo,
            "data_fidelity": L2(),
            "prior": prior,
            "get_output": custom_output,
            "backtracking": True,
            "early_stop": True,  # Stop algorithm when convergence criteria is reached
            "crit_conv": "cost",  # Convergence is reached when the difference of cost function between consecutive iterates is smaller than thres_conv
            "thres_conv": 1e-4,
        }
        return optim_builder(verbose=False, **optim_params)

    def init_dpir(self, prior_model, noise_level):
        prior = dinv.optim.prior.PnP(prior_model)
        sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level)
        optim_params = {
            "params_algo": {
                "stepsize": stepsize,
                "g_param": sigma_denoiser,
            },
            "iteration": "HQS",
            "prior": prior,
            "data_fidelity": L2(),
            "max_iter": max_iter,
            "backtracking": False,
            "early_stop": False,
        }
        return optim_builder(verbose=False, **optim_params)

    def init_dps(self, model, max_iter, eta, device):
        dps_model = DPSWrapper(
            model=model,
            data_fidelity=L2(),
            max_iter=max_iter,
            eta=eta,
            device=device,
            save_iterates=True,
        )
        return DinvReconstructorWrapper(dps_model)

    def init_diffpir(self, model, noise_level, max_iter=100, zeta=0.3, lambda_=6.0):
        diffpir_model = DiffPIR(
            model=model,
            data_fidelity=L2(),
            max_iter=max_iter,
            zeta=zeta,
            lambda_=lambda_,
            sigma=noise_level
        )
        return DinvReconstructorWrapper(diffpir_model)

    def init_wiener(self, balance):
        wiener_deconv = WienerSolver(balance=balance)
        return DinvReconstructorWrapper(wiener_deconv)

    def init_deep_prior(self, network: str, device, **kwargs):
        if network.lower() == "drunet":
            model = dinv.models.DRUNet(pretrained="download").to(device=device)
            distributed.print0("Initialized DRUNet prior")
        elif network.lower() == "diffunet":
            large_model = kwargs.get("large_model", False)
            model = dinv.models.DiffUNet(large_model=large_model, use_fp16=False, pretrained="download").to(device=device)
            distributed.print0(f"Initialized DiffUNet-{'large' if large_model else 'small'} prior")
        else:
            raise ValueError(f"Network '{network}' is not a valid deep prior.")
        return model

    def init_models(self, cfg, dsets, device):
        models = {}
        test_dset = dsets[Datasplit.TEST]
        assert test_dset is not None
        # Initialize perturbation model
        if "kernel" in cfg.models and (pert_path := cfg.models.kernel.get("path")) is not None:
            if not pert_path.endswith(".pkl"):
                raise ValueError(
                    f"Cannot load degradation model from '{pert_path}'. "
                    "The file must be a pickle (network-snapshot) file."
                )
            with open(pert_path, "rb") as fh:
                pretr_data = pickle.load(fh)
                kernel_nn = pretr_data["kernel_nn"]
            models["kernel_nn"] = kernel_nn.to(device)
        else:
            print("No kernel-NN loaded. Will use the default degradation!")
            models["kernel_nn"] = copy.deepcopy(test_dset.corruption).to(device)
            if hasattr(models["kernel_nn"], "device"):
                models["kernel_nn"].device = device

        # Initialize the DeepInv solver model
        noise_level = self.get_noise_level(test_dset, models["kernel_nn"])
        invp_cfg = cfg["models"]["deepinv_solver"]
        solver_method = invp_cfg["method"]
        if solver_method.lower() == "gspnp":
            prior = self.init_deep_prior("drunet", device)
            models["deepinv_model"] = self.init_gspnp(prior, noise_level)
        elif solver_method.lower() == "dpir":
            prior = self.init_deep_prior(invp_cfg["prior"], device, **invp_cfg)
            models["deepinv_model"] = self.init_dpir(prior, noise_level)
        elif solver_method.lower() == "dps":
            prior = self.init_deep_prior(invp_cfg["prior"], device, **invp_cfg)
            models["deepinv_model"] = self.init_dps(
                prior, max_iter=invp_cfg["max_iter"], eta=invp_cfg.get("eta", 1.0), device=device
            )
        elif solver_method.lower() == "diffpir":
            prior = self.init_deep_prior(invp_cfg["prior"], device, **invp_cfg)
            models["deepinv_model"] = self.init_diffpir(
                prior, noise_level=invp_cfg.get("sigma", noise_level), max_iter=invp_cfg.get("max_iter", 100),
                zeta=invp_cfg.get("zeta", 0.3), lambda_=invp_cfg.get("lambda_", 8.0)
            )
        elif solver_method.lower() == "wiener":
            models["deepinv_model"] = self.init_wiener(
                balance=invp_cfg.get("balance", noise_level)
            )
        else:
            raise ValueError(f"deepinv_solver method '{solver_method}' is invalid.")

        return models

    def get_physics(self, cfg) -> dinv.physics.Physics:
        # The physics is the one modeled and learned using DiffInstruct
        from ddm4ip.degradations.blur import Blur
        from ddm4ip.degradations.varpsf import PerPixelBlur
        from ddm4ip.degradations.downsampling import Downsampling
        kernel_nn = self.models["kernel_nn"]
        if self.physics is None:
            noise_sigma = self.get_noise_level(self.dsets[Datasplit.TEST], kernel_nn)
            noise_model = init_noise({"kind": "gaussian", "std": noise_sigma})

            if (deg_type := getattr(kernel_nn, "degradation_type", None)) is not None:
                if deg_type == BlurDegradationType.BLUR:
                    self.physics = Blur(
                        filter=None,
                        padding="replicate",  # Shouldn't be 'valid', or the complex and brittle patching code fails
                        noise_model=noise_model,
                        device=self.device  # type: ignore
                    )
                elif deg_type == BlurDegradationType.PER_PIXEL_BLUR:
                    self.physics = PerPixelBlur(
                        filters=None,
                        padding=kernel_nn.padding,
                        noise_model=noise_model,
                        device=self.device  # type: ignore
                    )
                elif deg_type == BlurDegradationType.DOWNSAMPLING:
                    dset = self.dsets[Datasplit.TEST]
                    assert dset is not None
                    self.physics = Downsampling(
                        filter=None,
                        padding="reflect",#kernel_nn.padding,
                        img_size=dset.clean_img_size,
                        device=self.device  # type: ignore
                    )
                else:
                    raise NotImplementedError(deg_type)
            elif isinstance(kernel_nn, dinv.physics.Physics):
                # path taken when doing baselines (true degradation kernels)
                # Adapter speeds up baseline when using per-pixel blurs
                # self.physics = SpaceToCenterAdapter(kernel_nn, img_h=128, img_w=128)
                self.physics = kernel_nn
            else:
                raise ValueError("Failed to initialize physics for the provided step-2 NN.")
            distributed.print0(f"Initialized physics: {self.physics}")

        return self.physics

    def get_noise_level(self, test_dset, kernel_model) -> float:
        if hasattr(kernel_model, "sigma"):
            print(f"Overriding provided sigma with sigma from model ({kernel_model.sigma.item()})")
            noise_level = kernel_model.sigma
        else:
            # The noise-level is assumed to be known, hence we just load the true
            # value from the dataset
            noise_level = test_dset.noise_level
        return noise_level.item() if isinstance(noise_level, torch.Tensor) else noise_level

    def init_loss(self, cfg, models) -> DeepInvLoss:
        return DeepInvLoss(
            cfg,
            self.get_physics(cfg),
            deepinv_model=models["deepinv_model"],
            kernel_nn=models["kernel_nn"]
        )

    def validate_batch(self, val_batch: Batch):
        metrics, pred_img, filters = self.loss_optim.val_loss_with_output(self, val_batch)
        if self.save_eval_to_file:
            assert self.eval_dir is not None and val_batch.corrupt is not None and val_batch.clean is not None
            if self.save_pred_only:
                img_batch = pred_img.cpu()
            elif val_batch.corrupt.shape == val_batch.clean.shape:
                img_batch = torch.cat([val_batch.corrupt.cpu(), pred_img.cpu(), val_batch.clean.cpu()], dim=-1)
            else:
                img_batch = torch.cat([pred_img.cpu(), val_batch.clean.cpu()], dim=-1)
            # Filters are just saved as pt, since it's too hard to convert them into good
            # looking images for all possible situations.
            torch.save(filters, os.path.join(self.eval_dir, f"kernel_{self.global_step:05d}.pt"))
            assert img_batch.dim() == 4
            for i in range(img_batch.shape[0]):
                write_img_pt(img_batch[i], os.path.join(self.eval_dir, f"img_{self.global_step + i:05d}.png"))
        return metrics

    def make_plots(self, cfg, dsets, dloaders, models, tb_writer, global_step):
        if self.plot_dloader is None:
            if (dl := dloaders.get(Datasplit.TEST)) is None:
                warnings.warn("Using training batch for plotting.")
                dl = dloaders[Datasplit.TRAIN]
                assert dl is not None
            self.plot_dloader = iter(dl)

        try:
            plot_batch = next(self.plot_dloader)
        except StopIteration:
            return

        models["deepinv_model"].eval()
        testset = dsets[Datasplit.TEST]
        assert testset is not None

        p1 = self.plot_generated(plot_batch, models["deepinv_model"])
        if distributed.get_rank() == 0 and p1 is not None:
            self.log_image(p1, 'denoiser/generated', tb_writer)

        p2 = None
        try:
            p2, p2_ker = plot_spacevarying_kernels(
                testset.corruption,
                self.loss_optim.patch_size or testset.corrupt_img_size[-1],
                self.models["kernel_nn"],
                self.batch_size,
                self.device
            )
        except TypeError:
            pass
        if distributed.get_rank() == 0 and p2 is not None:
            self.log_image(p2, 'denoiser/kernels', tb_writer)

    @torch.no_grad()
    def plot_generated(self, batch: Batch, deepinv_model, num_imgs: int = 16):
        dev_batch = batch[:num_imgs].to(self.device)
        if dev_batch.clean is None:
            return None
        assert dev_batch.corrupt is not None
        target = dev_batch.clean
        source = dev_batch.corrupt

        preds, _ = self.loss_optim.run_model(dev_batch)

        batch_size = target.shape[0]
        n_img_row = int(math.sqrt(batch_size))

        target = make_grid(
            target.clamp(0, 1),
            nrow=n_img_row, padding=0,
        ).permute(1, 2, 0).numpy(force=True)
        source = make_grid(
            source.clamp(0, 1),
            nrow=n_img_row, padding=0,
        ).permute(1, 2, 0).numpy(force=True)
        preds = make_grid(
            preds.clamp_(0, 1),
            nrow=n_img_row, padding=0,
        ).permute(1, 2, 0).numpy(force=True)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(6, 6))

        ax[0, 0].imshow(source, aspect='equal', interpolation='none')
        ax[0, 0].set_title(r"noisy")
        ax[0, 0].set_axis_off()

        ax[0, 1].imshow(target, aspect='equal', interpolation='none')
        ax[0, 1].set_title(r"clean")
        ax[0, 1].set_axis_off()

        ax[1, 0].imshow(preds, aspect='equal', interpolation='none')
        ax[1, 0].set_title(r"predicted")
        ax[1, 0].set_axis_off()

        diff = (target - preds).mean(2)  # average over channels
        title = r"errors"
        if np.abs(diff).max() < 0.1:
            diff *= 10
            title = rf"{title} (x10)"
        ax[1, 1].imshow(diff, aspect='equal', interpolation='none', cmap='RdBu', vmin=-1.0, vmax=1.0)
        ax[1, 1].set_title(title)
        ax[1, 1].set_axis_off()

        fig.tight_layout()
        return fig
