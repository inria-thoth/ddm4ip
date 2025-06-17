import argparse
import logging
import os.path
import sys
import pathlib
import scipy
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "DAN" / "codes"))
sys.path.append(str(pathlib.Path(__file__).parent.resolve() / "DAN" / "codes" / "config" / "DANv2"))

import DAN.codes.config.DANv2.options as option
from DAN.codes.config.DANv2.models import create_model
import DAN.codes.utils as util
from DAN.codes.data import create_dataloader, create_dataset


options = {
    "name": "DANx2_setting2",
    "model": "blind",
    "distortion": "sr",
    "scale": 2,
    "pca_matrix_path": "./DAN/pca_matrix/DANv2/pca_aniso_matrix_x2.pth",
    "dataset": {
        "name": "DIV2KRK",
        "mode": "LQGT",
        "scale": 2,
        "data_type": "img",  # or lmdb
        "dataroot_LQ": None,
        "dataroot_GT": None,
    },
    "network_G": {
        "which_model_G": "DAN",
        "setting": {
            "nf": 64,
            "nb": 10,
            "ng": 5,
            "input_para": 10,
            "loop": 4,
            "kernel_size": 11,
            "pca_matrix_path": "./DAN/pca_matrix/DANv2/pca_aniso_matrix_x2.pth",
            "upscale": 2,
        }
    },
    "path": {
        "pretrain_model_G": "./DAN/checkpoints/DANv2/danv2_x2_setting2.pth",
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
