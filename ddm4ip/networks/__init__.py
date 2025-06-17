import torch
from torch import nn


def init_kernel_net(
    cfg,
    img_size: tuple[int, int, int],
    cond_ch: int,
    label_dim: int = 0,
    is_output_noisy: bool = True
) -> nn.Module:
    """Initialize the 'kernel-net'. A function from image-space to image-space which approximates
    an unknown forward model.

    Args:
        cfg (_type_): configuration dictionary for the model
        img_res (int): resolution of input images
        img_ch (int): number of channels of input images
        cond_ch (int): number of channels of additional network input (a condition for generation)
        label_dim (int, optional): if greater than zero, conditional models (i.e. models which
            depend on a class-label) may be initialized. Note that not all kernel-nets support
            conditional modeling. Defaults to 0.
        is_output_noisy (bool, optional): Whether the outputs should have noise added to them.
            This flag allows to support both DiffInstruct on X (inputs are noisy images, outputs are
            clean images hence `is_output_noisy` should be False) and DiffInstruct on Y (inputs
            are clean images and outputs are noisy, hence `is_output_noisy` should be True).
            Defaults to True.

    Returns:
        nn.Module: A model with the following signature: `model(img, noise_level, class_labels=None)`
    """
    kernel_type = cfg['type']
    if kernel_type == "direct":
        from .blur_nets import DirectKernel
        if not is_output_noisy:
            raise ValueError("Cannot initialize `ConvKernel` with non-noisy output.")
        kernel_nn = DirectKernel(
            kernel_size=cfg['kernel_size'],
            padding=cfg.get('padding', 'replicate'),
            factor=cfg.get('factor', 1),
            sum_to_one=cfg.get("sum_to_one", True),
            learn_output_noise=cfg.get("learn_output_noise", False),
        )
    elif kernel_type == "conv":
        from .blur_nets import ConvKernel
        if not is_output_noisy:
            raise ValueError("Cannot initialize `ConvKernel` with non-noisy output.")
        kernel_nn = ConvKernel(
            kernel_size=cfg['kernel_size'],
            kernel_ch=cfg.get('kernel_ch', 1),
            padding=cfg.get('padding', 'replicate'),
            factor=cfg.get('factor', 1),
            sum_to_one=cfg.get("sum_to_one", True)
        )
    elif kernel_type == "conv_center":
        from .blur_nets import ConvCenterKernel
        kernel_nn = ConvCenterKernel(
            kernel_size=cfg['kernel_size'],
            kernel_ch=cfg.get('kernel_ch', 1),
            img_ch=img_size[0],
            cond_ch=cond_ch,
            img_res=img_size[-1],
            padding=cfg["padding"],
            learn_output_noise=cfg["learn_output_noise"],
            sum_to_one=cfg["sum_to_one"],
        )
    elif kernel_type == "direct_per_pixel":
        from .blur_nets import DirectPerPixelKernel
        kernel_nn = DirectPerPixelKernel(
            psf_path=cfg["psf_path"],
            kernel_size=cfg["kernel_size"],
            padding=cfg["padding"],
        )
    elif kernel_type == "direct_center":
        from .blur_nets import DirectCenterKernel
        kernel_nn = DirectCenterKernel(
            psf_path=cfg["psf_path"],
            kernel_size=cfg["kernel_size"],
            padding=cfg["padding"],
            num_psfs=cfg.get("num_psfs", None),
            kernel_ch=cfg.get("kernel_ch", None),
            sum_to_one=cfg["sum_to_one"],
            learn_output_noise=cfg["learn_output_noise"],
            initialization=cfg["initialization"]
        )
    elif kernel_type == "conv_sr":
        from .blur_nets import ConvSR
        kernel_nn = ConvSR(
            factor=cfg["factor"],
            padding=cfg["padding"],
            img_ch=img_size[0],
            img_res=img_size[-1],
            kernel_ch=cfg["kernel_ch"],
            kernel_size=cfg["kernel_size"],
            learn_output_noise=cfg.get("learn_output_noise", False),
        )
    else:
        raise ValueError(f"kernel type (config['models']['kernel']['type']) '{kernel_type}' is not valid.")
    # 'load_from' should specify a pretrained model path.
    if (prtr_path := cfg.get("load_from")) is not None:
        ckpt = torch.load(prtr_path, map_location="cpu", weights_only=False)
        if "ema" in ckpt:
            state = ckpt["ema"]["emas"][-1]
            print(f"Loading kernel from '{prtr_path}'. EMA state with std: {ckpt['ema']['stds'][-1]}")
        elif "flow_nn" in ckpt:
            # This is the output of `finetune_flow`
            state = ckpt["flow_nn"]["state_dict"]
            print(f"Loading kernel from '{prtr_path}'.")
        else:
            raise ValueError(
                f"Failed to load kernel-nn from '{prtr_path}'. The list of available keys "
                f"is {list(ckpt.keys())} but we're looking for 'ema' or 'flow_nn'."
            )
        kernel_nn.load_state_dict(state, strict=False)

    return kernel_nn