import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from diffusion.diffusion_utils import cosine_noise_schedule, save_diffusion_sample, to_torch, linear_noise_schedule
import clip
from clip.model import ModifiedResNet
from tool.utils import RequiresGradContext, mse, cosine_similarity
import torch.autograd as autograd

class ConditionalDiffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000, use_semantic=False,
                 initial_lr=2e-4, training_target='noise', noise_schedule='cosine',
                 auto_sample=False, sample_every_n_steps=1000):
        """
        Args:
            model (torch.nn.Module):
                The model used to predict noise for reverse diffusion.
            channels (int):
                The amount of input channels in each image.
            timesteps (int):
                The amount of timesteps used to generate the noising schedule.
            initial_lr (float):
                The initial learning rate for the diffusion training.
            training_target (str):
                The type of parameterization to train the backbone model on.
                Can be either 'x0' or 'noise'.
            noise_schedule (str):
                The type of noise schedule to be used.
                Can be either 'linear' or 'cosine'.
            auto_sample (bool):
                Should the model perform sampling during training.
                If False, the following sampling parameters are ignored.
            sample_every_n_steps (int):
                The amount of global steps (step == training batch) after which the model is
                sampled from.
        """
        super().__init__()

        self.step_counter = 0  # Overall step counter used to sample every n global steps
        self.sample_every_n_steps = sample_every_n_steps
        self.auto_sample = auto_sample
        self.use_semantic = use_semantic

        self.channels = channels
        self.model = model
        self.initial_lr = initial_lr
        self.training_target = training_target.lower()
        assert self.training_target in ['x0', 'noise']

        assert noise_schedule in ['linear', 'cosine']
        if noise_schedule == 'linear':
            betas = linear_noise_schedule(timesteps)
        else:
            betas = cosine_noise_schedule(timesteps)

        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])
        self.sqrt_alphas_hat_prev = np.sqrt(np.append(1., alphas_hat))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_hat', to_torch(alphas_hat))
        self.register_buffer('alphas_hat_prev', to_torch(alphas_hat_prev))
        self.register_buffer('sqrt_alphas_hat', to_torch(np.sqrt(alphas_hat)))
        self.register_buffer('sqrt_one_minus_alphas_hat', to_torch(np.sqrt(1. - alphas_hat)))
        self.register_buffer('log_one_minus_alphas_hat', to_torch(np.log(1. - alphas_hat)))
        self.register_buffer('sqrt_recip_alphas_hat', to_torch(np.sqrt(1. / alphas_hat)))
        self.register_buffer('sqrt_recipm1_alphas_hat', to_torch(np.sqrt(1. / alphas_hat - 1)))
        posterior_variance = betas * (1. - alphas_hat_prev) / (1. - alphas_hat)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_hat_prev) / (1. - alphas_hat)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_hat_prev) * np.sqrt(alphas) / (1. - alphas_hat)))

        if use_semantic:
            self.model_ex = CLIP_Semantic_extractor()
        else:
            self.model_ex = None

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_hat[t] * x_t - self.sqrt_recipm1_alphas_hat[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised, condition_x=None, frame_diff=None, img_semantic=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_hat_prev[t + 1]]).repeat(batch_size, 1).view((batch_size,)).to(x.device)

        if frame_diff is not None:
            fd_tensor = torch.full((batch_size,), frame_diff, dtype=torch.int64, device=self.device)
        else:
            fd_tensor = None

        if self.training_target == 'x0':
            x_recon = self.model(torch.cat([condition_x, x], dim=1), noise_level, fd_tensor, img_semantic=img_semantic)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(torch.cat([condition_x, x], dim=1),
                                                                             noise_level, fd_tensor, img_semantic=img_semantic))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def modified_p_mean_variance(self, x, t, clip_denoised, condition_x=None, frame_diff=None, img_semantic=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_hat_prev[t + 1]]).repeat(batch_size, 1).view((batch_size,)).to(x.device)

        if frame_diff is not None:
            fd_tensor = torch.full((batch_size,), frame_diff, dtype=torch.int64, device=self.device)
        else:
            fd_tensor = None

        if self.training_target == 'x0':
            x_recon = self.model(torch.cat([condition_x, x], dim=1), noise_level, fd_tensor, img_semantic=img_semantic)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(torch.cat([condition_x, x], dim=1),
                                                                             noise_level, fd_tensor, img_semantic=img_semantic))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon

    def p_sample(self, x, t, clip_denoised=True, condition_x=None, frame_diff=None, img_semantic=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, frame_diff=frame_diff, img_semantic=img_semantic)
        b, c, h, w = x.shape
        if t > 0:
            noise = torch.randn(size=(b, self.channels, h, w), device=x.device)
        else:
            noise = torch.zeros(size=(b, self.channels, h, w), device=x.device)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def modified_p_sample(self, x, t, clip_denoised=True, condition_x=None, frame_diff=None,
                          img_semantic=None, source_image=None, die=None,
                          ):
        model_mean, model_log_variance, x_recon = self.modified_p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, frame_diff=frame_diff, img_semantic=img_semantic)

        if source_image is None:
            source_image = condition_x
        if die is not None and t >= 0.98 * self.num_timesteps:
            assert source_image is not None, 'source_image is None'
            down, up = die
            with RequiresGradContext(x_recon, requires_grad=True):
                Y = up(down(x_recon))
                X = up(down(source_image))
                loss = 0.05 * mse(X, Y) + 0.05 * mse(source_image, x_recon)
                grad = autograd.grad(loss.sum(), x_recon)[0]
            model_mean = model_mean - grad.detach()

        b, c, h, w = x.shape
        if t > 0:
            noise = torch.randn(size=(b, self.channels, h, w), device=x.device)
        else:
            noise = torch.zeros(size=(b, self.channels, h, w), device=x.device)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def sample(self, condition, frame_diff=None, img_semantic=None):
        """
        Sample an image from noise via the reverse diffusion process, conditioned on several factors.
        Args:
            condition (torch.tensor):
                The conditioning tensor for the generation process.
            frame_diff (int):
                Used for DDPM frame predictor sampling. The frame index difference between the condition frame and
                the currently sampled frame. Can be None.
        """
        b, _, h, w = condition.shape
        img = torch.randn(size=(b, self.channels, h, w), device=condition.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=condition, frame_diff=frame_diff, img_semantic=img_semantic)
        return img

    def modified_sample(self, condition, frame_diff=None,
                        img_semantic=None, source_image=None, die=None, return_intermediate=False,
                        ):
        """
        Sample an image from noise via the reverse diffusion process, conditioned on several factors.
        Args:
            condition (torch.tensor):
                The conditioning tensor for the generation process.
            frame_diff (int):
                Used for DDPM frame predictor sampling. The frame index difference between the condition frame and
                the currently sampled frame. Can be None.
        """
        b, _, h, w = condition.shape
        img = torch.randn(size=(b, self.channels, h, w), device=condition.device)
        intermediate = [img]
        for i in reversed(range(0, self.num_timesteps)):
            img = self.modified_p_sample(img, i, condition_x=condition, frame_diff=frame_diff,
                                         img_semantic=img_semantic, die=die, source_image=source_image,
                                         )
            intermediate.append(img)
        if return_intermediate:
            return intermediate
        return img

    @torch.no_grad()
    def sample_ddim(self, condition, x_T=None, sampling_step_size=100, frame_diff=None):
        """
        Sample from the model, using the DDIM sampling process.
        The DDIM implicit sampling process is determinstic, and will always generate the same output
        if given the same input.

        Args:
            condition (torch.tensor):
                The image used to condition the sampling process.
            x_T (torch.tensor):
                The initial noise to start the sampling process from. Can be None.
            sampling_step_size (int):
                The step size between each t in the sampling process. The higher this value is, the faster the
                sampling process (as well as lower image quality).
            frame_diff (int):
                Used for DDPM frame predictor sampling. The frame index difference between the condition frame and
                the currently sampled frame. Can be None.
        """
        batch_size = condition.shape[0]
        seq = range(0, self.num_timesteps, sampling_step_size)
        seq_next = [-1] + list(seq[:-1])

        if frame_diff is not None:
            fd_tensor = torch.full((batch_size,), frame_diff, dtype=torch.int64, device=self.device)
        else:
            fd_tensor = None

        if x_T is None:
            x_t = torch.randn(condition.shape, device=self.device)
        else:
            x_t = x_T

        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))[:-1]
        for t, t_next in zipped_reversed_seq:
            noise_level = torch.FloatTensor(
                [self.sqrt_alphas_hat_prev[t + 1]]).repeat(batch_size, 1).view((batch_size,)).to(x_t.device)

            e_t = self.model(torch.cat([condition, x_t], dim=1), noise_level, frame_diff=fd_tensor)
            predicted_x0 = (x_t - self.sqrt_one_minus_alphas_hat[t] * e_t) / self.sqrt_alphas_hat[t]
            direction_to_x_t = self.sqrt_one_minus_alphas_hat[t_next] * e_t
            x_t = self.sqrt_alphas_hat[t_next] * predicted_x0 + direction_to_x_t

        t_tensor = torch.full((batch_size,), 0, dtype=torch.int64, device=self.device)
        e_t = self.model(torch.cat([condition, x_t], dim=1), t_tensor, fd_tensor)
        x_0 = (x_t - self.sqrt_one_minus_alphas_hat[0] * e_t) / self.sqrt_alphas_hat[0]
        return x_0

    def q_sample(self, x_start, continuous_sqrt_alpha_hat, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return continuous_sqrt_alpha_hat * x_start + (1 - continuous_sqrt_alpha_hat ** 2).sqrt() * noise

    def forward(self, x_in, noise=None, img_semantic=None):
        x_start = x_in['IMG']
        b = x_start.shape[0]
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_hat = torch.FloatTensor(np.random.uniform(
            self.sqrt_alphas_hat_prev[t - 1],
            self.sqrt_alphas_hat_prev[t],
            size=b
        )).to(x_start.device).view(b, -1)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start,
                                continuous_sqrt_alpha_hat=continuous_sqrt_alpha_hat.view(-1, 1, 1, 1),
                                noise=noise)

        recon = self.model(
            torch.cat([x_in['CONDITION_IMG'], x_noisy], dim=1), continuous_sqrt_alpha_hat.view(-1), x_in.get('FRAME'), img_semantic=img_semantic)

        if self.training_target == 'x0':
            return F.mse_loss(x_start, recon)
        else:
            return F.mse_loss(noise, recon)

    def training_step(self, batch, batch_idx):
        if self.use_semantic:
            with torch.no_grad():
                img_semantic = self.model_ex(batch.get('CONDITION_IMG'))
        else:
            img_semantic = None

        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            frame_diff = None if 'FRAME' not in batch else batch.get('FRAME').item()
            sample = self.sample(condition=batch['CONDITION_IMG'], frame_diff=frame_diff, img_semantic=img_semantic)
            if batch['CONDITION_IMG'].shape[1] == 3:
                # Used for predictor training
                save_diffusion_sample(batch['CONDITION_IMG'],
                                      os.path.join(self.logger.log_dir, f'{self.step_counter}_conditioning.png'))
            elif batch['CONDITION_IMG'].shape[1] == 6:
                # Used for interpolator training
                save_diffusion_sample(batch['CONDITION_IMG'][:, :3],
                                      os.path.join(self.logger.log_dir, f'{self.step_counter}_conditioning1.png'))
                save_diffusion_sample(batch['CONDITION_IMG'][:, 3:],
                                      os.path.join(self.logger.log_dir, f'{self.step_counter}_zconditioning2.png'))
            else:
                raise Exception(f'Condition channel count ({batch["CONDITION_IMG"].shape}) is not valid')
            save_diffusion_sample(sample, os.path.join(self.logger.log_dir, f'{self.step_counter}_sample.png'))

        loss = self.forward(batch, img_semantic=img_semantic)
        self.log('train_loss', loss)
        self.step_counter += 1
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20], gamma=0.1, verbose=True)
        return [optim], [scheduler]


class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path

        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')

        self.load_state_dict(model.visual.state_dict())
        # self.register_buffer(
        #     'mean',
        #     torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        # )
        # self.register_buffer(
        #     'std',
        #     torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        # )
        self.requires_grad_(False)

        del model

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # x = (x - self.mean) / self.std
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)  # /4
        x = self.layer1(x)
        x = self.layer2(x)  # /2
        # x = self.layer3(x)  # /2
        # x = self.layer4(x)  # /2

        return x
