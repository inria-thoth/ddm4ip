Clone restormer repo:
```bash
git clone https://github.com/swz30/Restormer.git
```

Then edit (or create a new) config file. An example is `ffhq_motiondeblur_config.yaml`, but more examples can be found in the Restormer repository.

Finally run
```bash
./train_restormer.sh <path-to-config-file>
```

> [!Tip]
> basicsr code (which is used for training restormer models) does not allow one to specify
> the output directory. As an example, given an experiment with name `"Deblurring_Restormer"`,
> the outputs will be located at `./home/gmeanti/inverseproblems/scripts/restormer/Restormer/experiments/Deblurring_Restormer`.
> It's important to create the output directory before training or model-saving will fail.
