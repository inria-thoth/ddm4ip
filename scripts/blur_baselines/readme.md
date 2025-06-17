# FFHQ Deblurring Experiments

The data used is [FFHQ](). Download the dataset and preprocess it using the `scripts/preprocess_ffhq.py` script.

For the baseline experiments only 1k images from the test set are used (cue the `ffhq_256_val.zip` created by the preprocessing script).

## [Fast Diffusion EM](https://arxiv.org/abs/2309.00287)

1. Clone the [repository](https://github.com/claroche-r/FastDiffusionEM) in the current folder.
2. Download the pretrained diffusion model checkpoint `"ffhq_10m.pt"` from the [DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling) at this [google drive link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)
3. Edit the `fastdiffem_config.yaml` file such that the `"model_path"` key points to the downloaded model
4. Create folder `mkdir FastDiffusionEM/model_zoo`. Download the motion-blur kernel diffusion model from [this google drive link](https://drive.google.com/drive/folders/1pueQC9FI0ozoSUiu4u1MlJR52zYSnvKr) and place it in the model_zoo folder.
5. Unzip the `ffhq_256_val.zip` somewhere. We'll assume it has been unzipped to `$DATA_DIR`. We will also assume there is some `$OUT_DIR` where results will be saved
6. Run Fast Diffusion EM:
    ```bash
    PYTHONPATH='.' python run_fastdiffem.py \
        --img_model_config=fastdiffem_config.yaml \
        --input_dir="${DATA_DIR}" \
        --save_dir="${OUT_DIR}" \
        --sigma=5 \
        --n=16 \
        --gpu=0
    ```
