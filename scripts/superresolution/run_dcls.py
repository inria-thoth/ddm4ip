import argparse
import logging
import os.path
import sys
import pathlib
import scipy
import torch
# TODO: There might be too many path appends here...
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "DCLS-SR"))
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "DCLS-SR" / "codes"))
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "DCLS-SR" / "codes" / "config" / "DCLS"))

# Patch DCLSv2 FFT code
import codes.utils.dcls_utils as dcls_utils

def new_rfft(x, dim, onesided):
    assert dim == 3
    assert not onesided
    x_fft = torch.fft.fftn(x, dim=(-3, -2, -1))
    return torch.stack((x_fft.real, x_fft.imag), -1)

def new_irfft(x, dim, onesided):
    assert dim == 3
    assert not onesided
    return torch.fft.ifft2(
        torch.complex(x[..., 0], x[..., 1]), dim=(-3, -2, -1)
    )

dcls_utils.torch.rfft = new_rfft
dcls_utils.torch.irfft = new_irfft

import codes.config.DCLS.options as option
from codes.config.DCLS.models import create_model
import codes.utils as util
from codes.data import create_dataloader, create_dataset


options = {
    "name": "DCLSx2_setting2",
    "model": "blind",
    "distortion": "sr",
    "scale": 2,
    "pca_matrix_path": "./DCLS-SR/pca_matrix/DCLS/pca_aniso_matrix_x2.pth",
    "dataset": {
        "name": "DIV2KRK",
        "mode": "LQGT",
        "scale": 2,
        "data_type": "img",  # or lmdb
        "dataroot_LQ": None,
        "dataroot_GT": None,
    },
    "network_G": {
        "which_model_G": "DCLS",
        "setting": {
            "nf": 64,
            "nb": 10,
            "ng": 5,
            "input_para": 256,
            "kernel_size": 11,
            "pca_matrix_path": "./DCLS-SR/pca_matrix/DCLS/pca_aniso_matrix_x2.pth",
            "upscale": 2,
        }
    },
    "path": {
        "pretrain_model_G": "./DCLS-SR/pretrained_models/DCLSx2_setting2.pth",
        "root": None,
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("--hq-path", type=str, required=True)
parser.add_argument("--lq-path", type=str, required=True)
parser.add_argument("--out-path", type=str, required=True)
args = parser.parse_args()

options["dataset"]["dataroot_LQ"] = args.lq_path
options["dataset"]["dataroot_GT"] = args.hq_path
options["path"]["root"] = args.out_path
options: dict = option.dict_to_nonedict(options) # type: ignore

#### mkdir and logger
util.mkdirs(options["path"]["root"])
util.setup_logger(
    "base",
    options["path"]["root"],
    "test_" + options["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(options))

#### Create test dataset and dataloader
test_set = create_dataset(options["dataset"])
test_loader = create_dataloader(test_set, options["dataset"])
logger.info(
    "Number of test images in [{:s}]: {:d}".format(
        options["dataset"]["name"], len(test_set)
    )
)

# load pretrained model by default
model = create_model(options)

test_set_name = test_loader.dataset.opt["name"]  # path opt['']
logger.info("\nTesting [{:s}]...".format(test_set_name))

for test_data in test_loader:
    single_img_psnr = []
    single_img_ssim = []
    single_img_psnr_y = []
    single_img_ssim_y = []
    need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
    img_path = test_data["LQ_path"][0]
    img_name = pathlib.Path(img_path).stem
    img_num = int(img_name.split("_")[1])

    #### input dataset_LQ
    model.feed_data(test_data["LQ"], test_data["GT"])
    model.test()
    visuals = model.get_current_visuals()
    SR_img = visuals["Batch_SR"]
    sr_img = util.tensor2img(visuals["SR"].squeeze())  # uint8
    ker = visuals['ker'].squeeze(0).reshape(11, 11)

    save_img_path = os.path.join(options["path"]["root"], img_name + ".png")
    save_ker_path = save_img_path.replace(".png", ".mat")
    print(f"Saving image {img_name} to {save_img_path}")
    util.save_img(sr_img, save_img_path)
    scipy.io.savemat(save_ker_path, {"Kernel": ker.numpy(force=True)})
