import argparse
import pickle
import scipy.io
import torch


def get_kernel(network_file: str, output_file: str):
    with open(network_file, "rb") as fh:
        pretr_data = pickle.load(fh)
        kernel_nn = pretr_data["kernel_nn"]
    if torch.cuda.is_available():
        kernel_nn = kernel_nn.cuda()
    else:
        kernel_nn = kernel_nn.cpu()
    kernel = kernel_nn.get_kernel(None, None)
    kernel = kernel.cpu().detach()
    kernel = kernel.reshape(kernel.shape[-2], kernel.shape[-1])
    kernel = kernel.numpy()
    scipy.io.savemat(output_file, {"Kernel": kernel})
    print(f"Saved {kernel.shape} kernel to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", "-n", help="Path to the network snapshot (.pkl file)")
    parser.add_argument("--output-file", "-o", help="Where to save the kernel (.mat file)")
    args = parser.parse_args()
    get_kernel(args.network, args.output_file)