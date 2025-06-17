# SuperResolution Experiments & baselines

The data used is the [DIV2KRK dataset](https://www.wisdom.weizmann.ac.il/~vision/kernelgan/) for all experiments.

All experiments are configured mostly through environment variables:
 - `DATA_DIR`: should point to the root of the unzipped DIV2KRK data
 - `OUTPUT_DIR`: should point to a directory where results should be saved

### Non-blind experiments

Clone the following repositories
```bash
git clone https://github.com/cszn/USRNet.git
git clone https://github.com/sefibk/KernelGAN.git  # for ZSSR
```
and use the following scripts
 - `run_zssr_baseline.sh` runs ZSSR with the true kernels.
 - `run_zssr_bicubic.sh` runs ZSSR with bicubic (x2) kernels.
 - `run_usrnet_baseline.sh` runs USRNet with the true kernels.

### DDM4IP

The main script is
 - `run_all_ddm4ip.sh` runs our algorithm with ZSSR reconstruction on the whole DIV2KRK, one image at a time.

but if you already trained the kernel-learning networks, you can also run
 - `run_zssr_ddm4ip.sh` runs ZSSR with a kernel learned with DDM4IP.
 - `run_usrnet_ddm4ip.sh` runs USRNet with learned kernels.
which additionally require setting env-vars `NETWORK_BASE_NAME` (the DDM4IP network should be in the same directory as `OUTPUT_DIR`, check the code for details on required file-structure) and `CHECKPOINT_STEP` from which to extract the kernel.

### KernelGAN

Clone kernelGAN
```bash
git clone https://github.com/sefibk/KernelGAN.git
```

and run
 - `run_kernelgan.sh` runs the kernelGAN baseline with ZSSR reconstruction.

### DIP-FKP

Clone FKP and USRNet
```bash
git clone https://github.com/JingyunLiang/FKP.git
git clone https://github.com/cszn/USRNet.git
```

Run using bash script
 - `run_dipfkp.sh` which will save results to directory `$OUTPUT_DIR/baseline_DIPFKP`

### DIP-DKP

Clone DKP and USRNet
```bash
git clone https://github.com/XYLGroup/DKP.git
git clone https://github.com/cszn/USRNet.git
```

> [!NOTE]
> You may need to modify the file at `DKP/DIPDKP/DIPDKp/model/model.py` to avoid using the `xlwt` (just commenting all usages of the library should be enough).

Run using bash script
 - `run_dipdkp.sh` which will save results to directory `$OUTPUT_DIR`


### DAN

Clone DAN and create checkpoints directory
```bash
git clone https://github.com/greatlog/DAN.git
mkdir DAN/checkpoints
```

Download pretrained checkpoints (see [links in github repository](https://github.com/greatlog/DAN)) to the `DAN/checkpoints` directory. Finally run on the DIV2KRK dataset with the bash script
 - `run_dan.sh` which will save results to directory `$OUTPUT_DIR/DANv2`

> [!NOTE]
> Running DAN with ground-truth or learned kernels requires modifying the DAN code.
> We aim to publish the required modifications soon, please open an issue if you need them.


### DCLSv2

```bash
git clone https://github.com/megvii-research/DCLS-SR.git
mkdir DCLS-SR/pretrained_models
```

Download pretrained checkpoints (see [links in github repository](https://github.com/megvii-research/DCLS-SR)) to the `DCLS-SR/pretrained_models` directory, and finally run the bash script
 - `run_dcls.sh` which will save results to directory `$OUTPUT_DIR/DCLS`

> [!IMPORTANT]
> The DCLSv2 code requires modifications to run with newer versions of pytorch. In `run_dcls.py` we attempt to patch the code to make it compatible, but this may not be very robust.