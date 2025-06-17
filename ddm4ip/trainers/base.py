
import abc
import copy
from enum import Enum
import os
import time

import dill as pickle
import re
from typing import Any, Dict, Mapping, MutableMapping, Optional, TypeAlias
import warnings

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.figure
from omegaconf import DictConfig
import psutil
from torch import Tensor
import torch.utils.data
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter

from ddm4ip.data.base import Datasplit, DatasetType, init_dataloader
from ddm4ip.utils import training_stats
from ddm4ip.utils import distributed
from ddm4ip.utils.easy_dict import print_config
from ddm4ip.utils.torch_utils import set_random_seed
from ddm4ip.utils.distributed import (
    get_local_rank,
    get_rank,
    get_world_size,
    print0,
    barrier,
)
from .train_utils import Timer, get_next_batch


TYPE_CFG = DictConfig
TYPE_DSETS = Mapping[Datasplit, DatasetType | None]
TYPE_DLOAD = Mapping[Datasplit, torch.utils.data.DataLoader | None]
TYPE_MODELS = MutableMapping[str, torch.nn.Module]
TYPE_LOSS: TypeAlias = 'losses.base.AbstractLoss'  # type: ignore # noqa: F821

class KEYS(str, Enum):
    TIME_DATA = "time/data"
    TIME_LOSS = "time/loss"
    TIME_MAINT = "time/maint"
    MEM_CPU = "resources/cpu_mem_gb"
    MEM_GPU = "resources/peak_gpu_mem_gb"


class AbstractTrainer(abc.ABC):
    @abc.abstractmethod
    def init_datasets(self, cfg: TYPE_CFG) -> TYPE_DSETS:
        pass

    @abc.abstractmethod
    def init_dataloaders(self, cfg: TYPE_CFG, dsets: TYPE_DSETS) -> TYPE_DLOAD:
        pass

    @abc.abstractmethod
    def init_models(self, cfg: TYPE_CFG, dsets: TYPE_DSETS, device) -> TYPE_MODELS:
        pass

    @abc.abstractmethod
    def init_loss(self, cfg: TYPE_CFG, models: TYPE_MODELS) -> TYPE_LOSS:
        pass

    @abc.abstractmethod
    def validate_batch(self, val_batch) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def make_plots(self,
                   cfg: TYPE_CFG,
                   dsets: TYPE_DSETS,
                   dloaders: TYPE_DLOAD,
                   models: TYPE_MODELS,
                   tb_writer: SummaryWriter,
                   global_step: int) -> None:
        pass

    @abc.abstractmethod
    def train(self, cfg):
        pass


class BaseTrainer(AbstractTrainer, abc.ABC):
    def __init__(self):
        self.is_training = False
        self.global_step = -1
        self.start_global_step = -1
        self.report_every_steps = -1
        self.plot_every_steps = -1
        self.save_every_steps = -1
        self.max_steps = -1
        self.batch_size = -1
        self.global_batch_size = -1
        self.max_val_batches = None
        self.save_eval_to_file = False
        self.save_first_step = False
        self.seed = 0
        self.run_dir: str | None = None
        self.ckpt_dir: str | None = None
        self.eval_dir: str | None = None
        self.device = torch.device("cpu")
        self.models = {}

    def init_valid_info(self, cfg):
        # Directories
        base_run_dir: str = cfg['training']['log_dir']
        self.run_dir = os.path.join(base_run_dir, cfg['exp_name'])
        self.eval_dir = os.path.join(self.run_dir, "plots")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        # Don't create ckpt_dir (it's only for loading)

        self.device = torch.device("cuda", get_local_rank())
        self.seed = cfg['training']['seed']
        set_random_seed(self.seed, get_rank())

        self.batch_size = cfg['training']['batch_size']
        if get_world_size() != 1:
            raise NotImplementedError("Validation with multiple GPUs is not implemented.")
        self.global_batch_size = self.batch_size * get_world_size()
        self.report_every_steps = cfg['training']['report_every_steps']
        if self.report_every_steps % self.global_batch_size != 0:
            raise ValueError(f"'report_every_steps' must be a multiple of batch_size, but is {self.report_every_steps}")
        self.plot_every_steps = cfg['training']['plot_every_steps']
        if self.plot_every_steps % self.global_batch_size != 0:
            raise ValueError(f"'plot_every_steps' must be a multiple of batch_size, but is {self.plot_every_steps}")
        if self.plot_every_steps % self.report_every_steps != 0:
            raise ValueError(
                f"'plot_every_steps' must be a multiple of 'report_every_steps'"
                f" but given {self.plot_every_steps} and {self.report_every_steps}"
            )
        self.max_steps = cfg['training'].get('max_steps')
        self.save_eval_to_file = cfg['training'].get('save_eval_to_file', False)
        self.save_pred_only = cfg['training'].get('save_pred_only', False)
        print0(f"Experiment:            {cfg['exp_name']}")
        print0(f"Seed:                  {self.seed}")
        print0(f"Will validate for:     {self.max_steps}")
        print0(f"Batch size is:         {self.batch_size}")
        print0(f"World size is:         {distributed.get_world_size()}")
        print0(f"Report every:          {self.report_every_steps} steps")
        print0(f"Plot every:            {self.plot_every_steps} steps")
        print0(f"Base directory:        {self.run_dir}")
        print0(f"Save plots to file:    {self.save_eval_to_file}")

    def init_train_info(self, cfg):
        # Directories
        base_run_dir: str = cfg['training']['log_dir']
        self.run_dir = os.path.join(base_run_dir, cfg['exp_name'])
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.eval_dir = os.path.join(self.run_dir, "plots")
        if get_rank() == 0:
            if os.path.exists(self.run_dir):
                warnings.warn(f"Experiment directory {self.run_dir} already exists.")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        self.device = torch.device("cuda", get_local_rank())
        self.seed = cfg['training']['seed']
        set_random_seed(self.seed, get_rank())

        self.batch_size = cfg['training']['batch_size']
        self.global_batch_size = self.batch_size * get_world_size() * cfg['loss'].get('n_accum_steps', 1)
        self.report_every_steps = cfg['training']['report_every_steps']
        if self.report_every_steps % self.global_batch_size != 0:
            raise ValueError(f"'report_every_steps' must be a multiple of {self.global_batch_size}, but is {self.report_every_steps}")
        self.plot_every_steps = cfg['training']['plot_every_steps']
        if self.plot_every_steps % self.global_batch_size != 0:
            raise ValueError(f"'plot_every_steps' must be a multiple of batch_size, but is {self.plot_every_steps}")
        if self.plot_every_steps % self.report_every_steps != 0:
            raise ValueError(
                f"'plot_every_steps' must be a multiple of 'report_every_steps'"
                f" but given {self.plot_every_steps} and {self.report_every_steps}"
            )
        self.save_every_steps = cfg['training']['save_every_steps']
        if self.save_every_steps % self.global_batch_size != 0:
            raise ValueError(f"'save_every_steps' must be a multiple of batch_size, but is {self.save_every_steps}")
        if self.save_every_steps % self.plot_every_steps != 0:
            raise ValueError(
                f"'save_every_steps' must be a multiple of 'plot_every_steps'"
                f" but given {self.save_every_steps} and {self.plot_every_steps}"
            )
        self.max_steps = cfg['training']['max_steps']
        self.max_val_batches = cfg['training'].get('max_val_batches', None)
        self.save_eval_to_file = cfg['training'].get('save_eval_to_file', False)
        self.save_first_step = cfg['training'].get('save_first_step', False)
        print0(f"Experiment:            {cfg['exp_name']}")
        print0(f"Seed:                  {self.seed}")
        print0(f"Will train for:        {self.max_steps}")
        print0(f"Per-GPU batch size is: {self.batch_size}")
        print0(f"Global batch size is:  {self.global_batch_size}")
        print0(f"World size is:         {distributed.get_world_size()}")
        print0(f"Report every:          {self.report_every_steps} steps")
        print0(f"Plot every:            {self.plot_every_steps} steps")
        print0(f"Save model every:      {self.save_every_steps} steps")
        print0(f"Max validation:        {self.max_val_batches} batches")
        print0(f"Base directory:        {self.run_dir}")
        print0(f"Checkpoints directory: {self.ckpt_dir}")
        print0(f"Save plots to file:    {self.save_eval_to_file}")
        print0(f"Save first step:       {self.save_first_step}")

    def maybe_load_checkpoint(self):
        assert self.ckpt_dir is not None

        start_global_step = 0
        # 1. Explicit checkpoint specified
        if (self.cfg['training'].get('checkpoint', None)) is not None:
            ckpt_path = self.cfg['training']['checkpoint']
        # 2. Checkpoint exists in the current experiment directory
        elif os.path.exists(self.ckpt_dir):
            # List available checkpoints
            pattern = r'training-state-(\d+).pt'
            ckpt_fnames = [
                entry.name for entry in os.scandir(self.ckpt_dir)
                if entry.is_file() and re.fullmatch(pattern, entry.name)
            ]
            if len(ckpt_fnames) == 0:
                return start_global_step
            ckpt_path = os.path.join(
                self.ckpt_dir,
                max(ckpt_fnames, key=lambda x: float(re.fullmatch(pattern, x).group(1))) # type: ignore
            )
        # 3. Checkpoint not present
        else:
            ckpt_path = None

        # Load the checkpoint.
        if ckpt_path is not None:
            ckpt_data = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            for k, v in self.models.items():
                self.models[k].load_state_dict(ckpt_data[k]['state_dict'])
            self.loss_optim.load_state_dict(ckpt_data['loss_optim'])
            start_global_step = ckpt_data['global_step']
            self.set_extra_state(ckpt_data)
            if start_global_step % self.global_batch_size != 0:
                raise ValueError(
                    f"Cannot load checkpoint with global step {start_global_step} "
                    f"which is not a multiple of batch size {self.global_batch_size}"
                )
            print0(f"Loaded checkpoint from           {ckpt_path}")
            print0(f"Will restart training from step  {start_global_step}")
        return start_global_step

    def should_report_at_step(self, step):
        return step != self.start_global_step and step % self.report_every_steps == 0

    def should_validate_at_step(self, step):
        return step != self.start_global_step and step % self.report_every_steps == 0

    def should_plot_at_step(self, step):
        if self.save_first_step and step == 0:
            return True
        return step != self.start_global_step and step % self.plot_every_steps == 0

    def should_save_at_step(self, step):
        if self.save_first_step and step == 0:
            return True
        return step != self.start_global_step and step % self.save_every_steps == 0

    def maybe_validate(self, val_dloader: Optional[torch.utils.data.DataLoader], val_loader_iter):
        if val_dloader is None:
            return val_loader_iter
        num_val_batches = len(val_dloader) if self.max_val_batches is None else self.max_val_batches
        if not self.should_validate_at_step(self.global_step):
            return val_loader_iter
        if num_val_batches <= 0:
            return val_loader_iter
        if not self.loss_optim.has_val_loss:
            return val_loader_iter
        for _ in range(num_val_batches):
            val_batch, val_loader_iter = get_next_batch(val_dloader, val_loader_iter)
            val_loss_info = self.validate_batch(val_batch)
            for k, v in val_loss_info.items():
                training_stats.report(k, v)
        return val_loader_iter

    def train_one_step(self, tr_dloader: torch.utils.data.DataLoader, tr_loader_iter, data_timer: Timer, loss_timer: Timer):
        while True:
            with data_timer.measure():
                train_batch, tr_loader_iter = get_next_batch(tr_dloader, tr_loader_iter)

            with loss_timer.measure():
                loss_info = self.loss_optim(self, train_batch)

            for k, v in loss_info.items():
                training_stats.report(k, v)

            if self.loss_optim.is_full_step_complete():
                break
        self.global_step += self.global_batch_size
        return tr_loader_iter


    def maybe_save_checkpoint(self):
        if self.should_save_at_step(self.global_step):
            if get_rank() == 0:
                assert self.ckpt_dir is not None
                save_file_name = os.path.join(self.ckpt_dir, f"training-state-{self.global_step}.pt")
                save_obj = {
                    k: {"state_dict": v.state_dict()}
                    for k, v in self.models.items()
                } | {
                    "loss_optim": self.loss_optim.state_dict(),
                    "global_step": self.global_step
                } | self.get_extra_state()
                torch.save(save_obj, save_file_name)
                print0(f"Saved checkpoint at '{save_file_name}'")

                save_obj = {}
                for mname, model in self.models.items():
                    try:
                        save_obj[mname] = copy.deepcopy(model).cpu().eval().requires_grad_(False)
                    except:
                        print0(mname)
                        print0(model)
                        raise
                save_obj["global_step"] = self.global_step + 1
                save_obj["run_dir"] = self.run_dir
                save_file_name = os.path.join(self.ckpt_dir, f"network-snapshot-{self.global_step}.pkl")
                with open(save_file_name, 'wb') as f:
                    pickle.dump(save_obj, f)
                print0(f"Saved network snapshot at '{save_file_name}'")
            barrier()

    def log_image(self, img: Tensor | matplotlib.figure.Figure, name: str, tb: Optional[SummaryWriter]):
        assert self.eval_dir is not None
        gs = self.global_step

        if isinstance(img, torch.Tensor):
            if tb is not None:
                tb.add_image(name, img, global_step=gs)
            if self.save_eval_to_file:
                fn = os.path.join(self.eval_dir, f"{name.replace('/', '_')}-{gs}.png")
                if img.dtype == torch.float32:
                    img = (img * 255).to(dtype=torch.uint8)
                torchvision.io.write_png(img, fn, 9)
        else:
            assert isinstance(img, matplotlib.figure.Figure)
            if tb is not None:
                tb.add_figure(name, img, global_step=gs, close=not self.save_eval_to_file)
            if self.save_eval_to_file:
                fn = os.path.join(self.eval_dir, f"{name.replace('/', '_')}-{gs}.png")
                img.savefig(fn, dpi=120, bbox_inches='tight')
                plt.close(img)

    def collect_stats(
        self,
        tb: Optional[SummaryWriter],
        collector=training_stats.default_collector,
        print_losses: bool = False,
        key_prefix: str = "",
    ):
        collector.update()
        stats_dict = dict(sorted(collector.as_dict().items()))
        stats_strings = []
        for k, v in stats_dict.items():
            if 'time' in k or 'resources' in k:
                value = v['mean']
                fmt_str = "5.1f"
            else:
                value = v['mean']
                fmt_str = "10.3e"
            if tb is not None:
                tb.add_scalar(f"{key_prefix}{k}", value, global_step=self.global_step)
            if k.startswith("val_loss") or k.startswith("loss") or k.startswith("learning_rate"):
                if print_losses:
                     stats_strings.append(f"{key_prefix}{k} {value:{fmt_str}}")
            else:
                stats_strings.append(f"{key_prefix}{k} {value:{fmt_str}}")
        print0(f"[{time.strftime('%d/%m %H:%M:%S', time.localtime())}][{self.global_step:7d}] {' | '.join(stats_strings)}")

    def evaluate(self, cfg):
        self.is_training = False
        self.cfg = cfg
        if get_rank() == 0:
            print0("-------------------------CONFIGURATION-------------------------")
            print_config(cfg)
            print0("---------------------------------------------------------------")

        self.init_valid_info(self.cfg)
        self.start_global_step = 0
        print0()
        self.dsets = self.init_datasets(cfg)
        if self.dsets.get(Datasplit.TEST, None) is None:
            raise KeyError("'test' dataset was not found, but it is mandatory.")
        self.models = self.init_models(cfg, self.dsets, self.device)
        self.loss_optim = self.init_loss(cfg, self.models)
        tb = SummaryWriter(log_dir=self.run_dir)

        self.maybe_load_checkpoint()
        self.dloaders = self.init_dataloaders(cfg, self.dsets)
        if (val_dloader := self.dloaders.get(Datasplit.TEST)) is None:
            raise KeyError("'test' dataloader was not found, but it is mandatory.")
        print("Initializing validation iterator")
        val_loader_iter = iter(val_dloader)
        print0("---------------------------------------------------------------")

        global_collector = training_stats.Collector(regex=r".*loss.*")  # only losses, no resources
        valid_timer = Timer()
        self.global_step = 0
        while True:
            if self.should_report_at_step(self.global_step):  # Report statistics to tensorboard and print
                training_stats.report(KEYS.TIME_MAINT.value, valid_timer.cuda_time())
                training_stats.report(KEYS.MEM_CPU.value, psutil.Process(os.getpid()).memory_info().rss / 2**30)
                training_stats.report(KEYS.MEM_GPU.value, torch.cuda.max_memory_allocated(self.device) / 2**30)
                torch.cuda.reset_peak_memory_stats()
                self.collect_stats(tb)

            if self.should_plot_at_step(self.global_step):
                with torch.no_grad():
                    self.make_plots(
                        cfg, self.dsets, self.dloaders, self.models, tb, self.global_step
                    )

            for k, v in self.models.items():
                v.eval()
            try:
                val_batch = next(val_loader_iter)
            except (StopIteration, RuntimeError):
                print0(f"Validation finished at step {self.global_step} due to: validation loader exhausted")
                break
            with valid_timer.measure():  # Validation loss
                val_loss_info = self.validate_batch(val_batch)
                for k, v in val_loss_info.items():
                    training_stats.report(k, v)
                self.global_step += val_batch.batch_size

            if self.max_steps is not None and self.global_step >= self.max_steps:
                print0(f"Validation finished at step {self.global_step}")
                break

        # Collect global/overall stats (averaged over the whole dataset.)
        self.collect_stats(
            tb,
            collector=global_collector,
            print_losses=True,
            key_prefix="overall_"
        )

    def train(self, cfg):
        self.is_training = True
        self.cfg = cfg
        if get_rank() == 0:
            print0("-------------------------CONFIGURATION-------------------------")
            print_config(cfg)
            print0("---------------------------------------------------------------")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        self.init_train_info(self.cfg)
        print0()
        self.dsets = self.init_datasets(cfg)
        if self.dsets.get(Datasplit.TRAIN, None) is None:
            raise KeyError("'train' dataset was not found, but it is mandatory.")

        self.models = self.init_models(cfg, self.dsets, self.device)

        self.loss_optim = self.init_loss(cfg, self.models)
        tb = SummaryWriter(log_dir=self.run_dir)
        self.tb_writer = tb

        self.start_global_step = self.maybe_load_checkpoint()
        if self.start_global_step >= self.max_steps + 1:
            raise ValueError(
                f"Given a starting step {self.start_global_step} "
                f"higher than the max number of steps {self.max_steps + 1}"
            )

        self.dloaders = self.init_dataloaders(cfg, self.dsets)
        if self.dloaders.get(Datasplit.TRAIN, None) is None:
            raise KeyError("'train' dataloader was not found, but it is mandatory.")
        print0("---------------------------------------------------------------")

        tr_dloader: torch.utils.data.DataLoader = self.dloaders[Datasplit.TRAIN] # type: ignore
        tr_loader_iter = iter(tr_dloader)
        val_dloader = self.dloaders.get(Datasplit.TEST)
        val_loader_iter = None
        if val_dloader is not None:
            val_loader_iter = iter(val_dloader)

        loss_timer = Timer()
        data_timer = Timer()
        valid_timer = Timer()
        eval_timer = Timer(cpu_only=True)
        save_timer = Timer(cpu_only=True)

        torch.set_anomaly_enabled(False)
        self.global_step = self.start_global_step
        while True:
            for k, v in self.models.items():
                v.eval()
            with valid_timer.measure():  # Validation loss
                self.maybe_validate(val_dloader, val_loader_iter)
            if self.should_report_at_step(self.global_step):  # Report statistics to tensorboard and print
                training_stats.report(KEYS.TIME_DATA.value, data_timer.cuda_time())
                training_stats.report(KEYS.TIME_LOSS.value, loss_timer.cuda_time())
                training_stats.report(KEYS.TIME_MAINT.value, valid_timer.cuda_time() + eval_timer.cpu_time() + save_timer.cpu_time())
                training_stats.report(KEYS.MEM_CPU.value, psutil.Process(os.getpid()).memory_info().rss / 2**30)
                training_stats.report(KEYS.MEM_GPU.value, torch.cuda.max_memory_allocated(self.device) / 2**30)
                torch.cuda.reset_peak_memory_stats()
                self.collect_stats(tb)
            with eval_timer.measure():
                if self.should_plot_at_step(self.global_step):
                    with torch.no_grad():
                        self.make_plots(
                            cfg, self.dsets, self.dloaders, self.models, tb, self.global_step
                        )
            with save_timer.measure():
                self.maybe_save_checkpoint()

            for k, v in self.models.items():
                v.train()
            tr_loader_iter = self.train_one_step(
                tr_dloader,
                tr_loader_iter,
                data_timer,
                loss_timer
            )
            self.on_train_step_finished()

            if self.global_step >= self.max_steps:
                print0(f"Training finished at step {self.global_step}")
                break

    def on_train_step_finished(self):
        pass

    def get_extra_state(self) -> Dict[str, Any]:
        return {}

    def set_extra_state(self, state: Dict[str, Any]):
        pass


def init_trts_dataloaders(cfg, dsets, start_idx, **kwargs):
    seed = cfg['training']['seed']
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training'].get('num_workers', 2)
    dataloaders = {}
    for split, dset in dsets.items():
        if dset is None:
            continue
        dataloaders[split] = init_dataloader(
            dset=dset,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            start_idx=start_idx,
            **kwargs
        )
    return dataloaders
