"""
Tidy up the DiffPIR implementation of deepinv (mainly to fix a device-placement bug)
"""
import math
import torch
from tqdm import tqdm
from deepinv.models import Reconstructor
import deepinv.physics


class DiffPIR(Reconstructor):
    def __init__(
        self,
        model,
        data_fidelity,
        sigma=0.05,
        max_iter=100,
        zeta=0.1,
        lambda_=7.0,
        verbose=False,
    ):
        super().__init__()
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity
        self.max_iter = max_iter
        self.zeta = zeta
        self.verbose = verbose
        self.beta_start, self.beta_end = 0.1 / 1000, 20 / 1000
        self.num_train_timesteps = 1000

        (
            self.sqrt_1m_alphas_cumprod,
            self.reduced_alpha_cumprod,
            self.sqrt_alphas_cumprod,
            self.sqrt_recip_alphas_cumprod,
            self.sqrt_recipm1_alphas_cumprod,
            self.betas,
        ) = self.get_alpha_beta()

        self.rhos, self.sigmas, self.seq = self.get_noise_schedule(sigma=sigma)

    def get_alpha_beta(self):
        """
        Get the alpha and beta sequences for the algorithm. This is necessary for mapping noise levels to timesteps.
        """
        betas = torch.linspace(
            self.beta_start, self.beta_end, self.num_train_timesteps
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        reduced_alpha_cumprod = torch.div(
            sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
        )  # equivalent noise sigma on image
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        return (
            sqrt_1m_alphas_cumprod,
            reduced_alpha_cumprod,
            sqrt_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            betas,
        )

    def get_noise_schedule(self, sigma):
        sigmas = self.reduced_alpha_cumprod.flip(0)
        sigma_ks = self.sqrt_1m_alphas_cumprod / self.sqrt_alphas_cumprod
        rhos = self.lambda_ * (sigma ** 2) / (sigma_ks ** 2)

        seq = torch.sqrt(torch.linspace(0, self.num_train_timesteps**2, self.max_iter))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1

        return rhos, sigmas, seq

    def find_nearest(self, array, value):
        return torch.abs(array - value).argmin()

    def forward(
        self,
        y,
        physics: deepinv.physics.LinearPhysics,
        seed=None,
        x_init=None,
    ):
        r"""
        Runs the diffusion to obtain a random sample of the posterior distribution.

        :param torch.Tensor y: the measurements.
        :param deepinv.physics.LinearPhysics physics: the physics operator.
        :param float sigma: the noise level of the data.
        :param int seed: the seed for the random number generator.
        :param torch.Tensor x_init: the initial guess for the reconstruction.
        """

        if seed:
            torch.manual_seed(seed)

        if hasattr(physics.noise_model, "sigma"):
            sigma = physics.noise_model.sigma  # Then we overwrite the default values
            self.rhos, self.sigmas, self.seq = self.get_noise_schedule(sigma=sigma.item())

        # Initialization
        if x_init is None:  # Necessary when x and y don't live in the same space
            x = 2 * physics.A_adjoint(y) - 1
        else:
            x = 2 * x_init - 1

        with torch.no_grad():
            for i in tqdm(range(len(self.seq)), disable=(not self.verbose)):

                # Current noise level
                curr_sigma = self.sigmas[self.seq[i]]

                # time step associated with the noise level sigmas[i]
                t_i = self.find_nearest(
                    self.reduced_alpha_cumprod, curr_sigma
                )

                if (i == 0):  # Initialization (simpler than the original code, may be suboptimal)
                    # x = self.sqrt_alphas_cumprod[t_i] * x + self.sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)
                    x = (
                        x + curr_sigma * torch.randn_like(x)
                    ) / self.sqrt_recip_alphas_cumprod[t_i]

                # Denoising step
                x_aux = (x * self.sqrt_recip_alphas_cumprod[t_i]) / 2 + 0.5  # renormalize in [0, 1]
                out = self.model(x_aux, curr_sigma / 2)
                denoised = 2 * out - 1
                x0 = denoised.clamp(-1, 1)

                if self.seq[i] != self.seq[-1]:
                    # Data fidelity step
                    x0_p = x0 / 2 + 0.5
                    x0_p = self.data_fidelity.prox(
                        x0_p, y, physics, gamma=1.0 / (2 * self.rhos[t_i])
                    )
                    x0 = x0_p * 2 - 1

                    # Sampling step
                    t_im1 = self.find_nearest(
                        self.reduced_alpha_cumprod,
                        self.sigmas[self.seq[i + 1]]
                    )  # time step associated with the next noise level

                    eps = (
                        x - self.sqrt_alphas_cumprod[t_i] * x0
                    ) / self.sqrt_1m_alphas_cumprod[t_i]  # effective noise

                    x = (
                        self.sqrt_alphas_cumprod[t_im1] * x0
                        + self.sqrt_1m_alphas_cumprod[t_im1]
                        * math.sqrt(1 - self.zeta)
                        * eps
                        + self.sqrt_1m_alphas_cumprod[t_im1]
                        * math.sqrt(self.zeta)
                        * torch.randn_like(x)
                    )  # sampling

        out = x0 / 2 + 0.5  # back to [0, 1] range

        return out

