import numpy as np
import yaml
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from common_utils.common import two_tuple
from diffusion.diffusion_utils import save_diffusion_sample, to_torch, linear_noise_schedule, cosine_noise_schedule
from tool.utils import dict2namespace
import clip
from clip.model import ModifiedResNet
from tool.utils import RequiresGradContext, mse, cosine_loss, compose_text_with_templates
import torch.autograd as autograd
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class Diffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000, use_semantic=False,
                 initial_lr=2e-4, training_target='x0', noise_schedule='cosine',
                 auto_sample=False, sample_every_n_steps=1000, sample_size=(32, 32),
                 ):
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
            sample_size (tuple):
                The spatial dimensions of the sample during auto sampling.
        """
        super().__init__()

        self.step_counter = 0  # Overall step counter used to sample every n global steps
        self.auto_sample = auto_sample
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_size = sample_size
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

        self.num_timesteps = int(betas.shape[0])

        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])

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
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised, img_semantic=None):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)

        if self.training_target == 'x0':
            x_recon = self.model(x, t_tensor, img_semantic=img_semantic)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t_tensor, img_semantic=img_semantic))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def modified_p_mean_variance(self, x, t, clip_denoised, img_semantic=None):
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)

        if self.training_target == 'x0':
            x_recon = self.model(x, t_tensor, img_semantic=img_semantic)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t_tensor, img_semantic=img_semantic))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, img_semantic=None):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised,
                                                                 img_semantic=img_semantic)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # no noise when t == 0
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def modified_p_sample(self, x, t, clip_denoised=True, img_semantic=None, source_image=None, die=None,
                          t2l_clip_extractor=None, text_embedds=None, ts=0):
        model_mean, _, model_log_variance, x_recon = self.modified_p_mean_variance(x=x, t=t,
                                                                                   clip_denoised=clip_denoised,
                                                                                   img_semantic=img_semantic)

        if die is not None and t >= ts:
            assert source_image is not None, 'source_image is None'
            down, up = die
            with RequiresGradContext(x_recon, requires_grad=True):
                Y = up(down(x_recon))
                X = up(down(source_image))
                loss = 0.05 * mse(X, Y) + 0.05 * mse(source_image, x_recon)
                grad = autograd.grad(loss.sum(), x_recon)[0]
            model_mean = model_mean - grad.detach()

        if t2l_clip_extractor is not None and text_embedds is not None:
            with RequiresGradContext(x_recon, requires_grad=True):
                x_recon_renorm = (x_recon + 1) * 0.5
                score = t2l_clip_extractor.calculate_clip_loss(x_recon_renorm, text_embedds)
                clip_grad = autograd.grad(score, x_recon, create_graph=False)[0]
            model_mean = model_mean - 500 * clip_grad.detach()

        # if dist_image is not None:
        #     assert die is not None, 'die is None'
        #     down, up = die
        #     model_mean = model_mean - up(down(model_mean)) + up(down(self.q_sample(dist_image, t)))

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)  # no noise when t == 0
        return model_mean + noise * (0.5 * model_log_variance).exp(), x_recon

    @torch.no_grad()
    def ddim_reverse_sample(self, x_t, custom_timesteps, clip_denoised=True, img_semantic=None,
                            return_intermediate=False):
        # x_t = x_t * 2 - 1
        timesteps = custom_timesteps or self.num_timesteps
        seq = range(0, timesteps)
        seq_next = list(seq[1:]) + [-1]

        batch_size = x_t.shape[0]
        zipped_reversed_seq = list(zip(seq, seq_next))
        intermediate = [x_t]
        for t, t_next in zipped_reversed_seq[:-1]:
            t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
            if self.training_target == 'x0':
                x_recon = self.model(x_t, t_tensor, img_semantic=img_semantic)
                eps = (x_t - self.sqrt_alphas_hat[t] * x_recon) / self.sqrt_one_minus_alphas_hat[t]
            else:
                eps = self.model(x_t, t_tensor, img_semantic=img_semantic)
                x_recon = self.predict_start_from_noise(x_t, t=t, noise=eps)
            if clip_denoised:
                x_recon.clamp_(-1., 1.)

            x_t = self.sqrt_alphas_hat[t_next] * x_recon + self.sqrt_one_minus_alphas_hat[t_next] * eps
            intermediate.append(x_t)
        if return_intermediate:
            return intermediate
        return x_t

    @torch.no_grad()
    def sample(self, image_size=(32, 32), batch_size=16, custom_initial_img=None, custom_timesteps=None,
               img_semantic=None):
        """
        Sample an image from noise via the reverse diffusion process.
        Args:
            image_size (int or tuple(int, int)):
                Spatial size of image to sample.
            batch_size (int):
                Amount of images to sample.
            custom_initial_img (torch.tensor):
                A non-default image to start the reverse process from. If this parameter is specified, both image_size
                and batch_size parameters are ignored.
                This can be used for denoising of partially noised images.
            custom_timesteps (int):
                A non-default diffusion timesteps parameter. If this parameter is specified, the reverse process is
                iterated for this given number of steps. Otherwise, the timestep parameter configured in the constructor
                is used.
                This can be used for denoising of partially noised images.
        """
        image_size = two_tuple(image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        timesteps = custom_timesteps or self.num_timesteps

        img = custom_initial_img if custom_initial_img is not None else torch.randn(sample_shape, device=self.device)

        # x_rand = torch.ones_like(img)
        # x_rand.normal_()
        # img = img + x_rand * 0.9

        # img = self.sample_ddim(x_T=img, sampling_step_size=1, img_semantic=img_semantic,
        #                        star_timesteps=int(0.8 * timesteps)
        #                        )

        for t in reversed(range(0, int(1 * timesteps))):
            img = self.p_sample(img, t, img_semantic=img_semantic)
        return img

    def modified_sample(self, image_size=(32, 32), batch_size=16, custom_initial_img=None, custom_timesteps=None,
                        img_semantic=None, source_image=None, die=None, return_intermediate=False,
                        t2l_clip_extractor=None, text_embedds=None, ts=0, return_x_recon=False,
                        ):
        """
        Sample an image from noise via the reverse diffusion process.
        Args:
            image_size (int or tuple(int, int)):
                Spatial size of image to sample.
            batch_size (int):
                Amount of images to sample.
            custom_initial_img (torch.tensor):
                A non-default image to start the reverse process from. If this parameter is specified, both image_size
                and batch_size parameters are ignored.
                This can be used for denoising of partially noised images.
            custom_timesteps (int):
                A non-default diffusion timesteps parameter. If this parameter is specified, the reverse process is
                iterated for this given number of steps. Otherwise, the timestep parameter configured in the constructor
                is used.
                This can be used for denoising of partially noised images.
        """
        image_size = two_tuple(image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        timesteps = custom_timesteps or self.num_timesteps

        img = custom_initial_img if custom_initial_img is not None else torch.randn(sample_shape, device=self.device)

        intermediate = [img]
        for t in reversed(range(0, int(1 * timesteps))):
            img, x_recon = self.modified_p_sample(img, t, img_semantic=img_semantic, source_image=source_image, die=die,
                                                  t2l_clip_extractor=t2l_clip_extractor, text_embedds=text_embedds,
                                                  ts=ts)
            if return_x_recon:
                intermediate.append(x_recon)
            intermediate.append(img)

        if return_intermediate:
            return intermediate
        return img

    @torch.no_grad()
    def sample_ddim(self, x_T=None, image_size=(32, 32), batch_size=16, sampling_step_size=100,
                    img_semantic=None, star_timesteps=0, return_intermediate=False
                    ):
        """
        Sample from the model, using the DDIM sampling process.
        The DDIM implicit sampling process is determinstic, and will always generate the same output
        if given the same input.

        Args:
            x_T (torch.tensor): The initial noise to start the sampling process from. Can be None.
            image_size (int or tuple(int)): Used as the sample spatial dimensions in case x_T is None.
            batch_size (int): Used as the sample batch size in case x_T is None.
            sampling_step_size (int): The step size between each t in the sampling process. The higher this value is,
                                      the faster the sampling process (as well as lower image quality).
        """
        seq = range(star_timesteps, self.num_timesteps, sampling_step_size)
        seq_next = [-1] + list(seq[:-1])

        if x_T is None:
            image_size = two_tuple(image_size)
            sample_shape = (batch_size, self.channels, image_size[0], image_size[1])
            x_t = torch.randn(sample_shape, device=self.device)
        else:
            batch_size = x_T.shape[0] if len(x_T.shape) == 4 else 1
            x_t = x_T

        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))
        intermediate = [x_t]
        for t, t_next in zipped_reversed_seq[:-1]:
            t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
            x_recon = self.model(x_t, t_tensor, img_semantic=img_semantic)

            # if clip_denoised:
            #     x_recon.clamp_(-1., 1.)
            if t > 0:
                e_t = (x_t - self.sqrt_alphas_hat[t] * x_recon) / self.sqrt_one_minus_alphas_hat[t]
                direction_to_x_t = self.sqrt_one_minus_alphas_hat[t_next] * e_t
                x_t = self.sqrt_alphas_hat[t_next] * x_recon + direction_to_x_t
            else:
                x_t = x_recon
            intermediate.append(x_t)

        if return_intermediate:
            return intermediate
        return x_t

    def q_sample(self, x_start, t, noise=None):
        """
        Perform forward diffusion (noising) in a single step.
        This method returns x_t, which is x_0 noised for t timesteps.

        Args:
            x_start (torch.Tensor): Represents the original image (x_0).
            t (int): The timestep that measures the amount of noise to add.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.sqrt_alphas_hat[t] * x_start + self.sqrt_one_minus_alphas_hat[t] * noise

    def forward(self, x, img_semantic, std=0.1, *args, **kwargs):
        x = x.get('IMG')

        batch_size = x.shape[0]

        # Sample t uniformly
        t = np.random.randint(0, self.num_timesteps)

        # Generate white noise
        noise = torch.randn_like(x)

        # Produce x_t (noisy version of input after t noising steps)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)

        # Attempt to reconstruct white noise that was used in forward process
        t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
        if self.training_target == 'x0':
            x0_recon = self.model(x_noisy, t_tensor, img_semantic=img_semantic)
            return F.mse_loss(x, x0_recon)
        else:
            noise_recon = self.model(x_noisy, t_tensor, img_semantic=img_semantic)
            return F.mse_loss(noise, noise_recon)

    def training_step(self, batch, batch_idx):
        sample_size = self.sample_size
        if batch.get('sample_size') is not None:
            sample_size = batch.get('sample_size')
        if self.use_semantic:
            with torch.no_grad():
                img_semantic = self.model_ex(batch.get('IMG'))
        else:
            img_semantic = None

        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            sample = self.sample(image_size=sample_size, batch_size=1, img_semantic=img_semantic)
            save_diffusion_sample(sample, f'{self.logger.log_dir}/sample_{self.step_counter}.jpg')

        loss = self.forward(batch, img_semantic=img_semantic)
        self.log('train_loss', loss)
        self.step_counter += 1
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20], gamma=0.1, verbose=True)
        return [optim], [scheduler]


class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32,
                 clip_affine_transform_fill=True, n_aug=16):
        global model
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path

        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')

        self.load_state_dict(model.visual.state_dict())
        self.requires_grad_(False)

        del model

        # self.model = model.eval().requires_grad_(False)
        # self.text_criterion = cosine_loss
        # self.clip_input_size = 224
        # self.clip_normalize = T.Normalize(
        #     mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        # )
        # self.basic_transform = T.Compose(
        #     [
        #         # we added interpolation to CLIP positional embedding, allowing to work with arbitrary resolution.
        #         T.Resize(self.clip_input_size, max_size=380),
        #         self.clip_normalize,
        #     ]
        # )
        # # list of augmentations we apply before calculating the CLIP losses
        # self.augs = T.Compose(
        #     [
        #         T.RandomHorizontalFlip(p=0.5),
        #         T.RandomApply(
        #             [
        #                 T.RandomAffine(
        #                     degrees=15,
        #                     translate=(0.1, 0.1),
        #                     fill=clip_affine_transform_fill,
        #                     interpolation=InterpolationMode.BILINEAR,
        #                 )
        #             ],
        #             p=0.8,
        #         ),
        #         T.RandomPerspective(
        #             distortion_scale=0.4,
        #             p=0.5,
        #             interpolation=InterpolationMode.BILINEAR,
        #             fill=clip_affine_transform_fill,
        #         ),
        #         T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.7),
        #         T.RandomGrayscale(p=0.15),
        #     ]
        # )
        #
        # self.n_aug = n_aug

    # def augment_input(self, input, n_aug=None, clip_input_size=None):
    #     if n_aug is None:
    #         n_aug = self.n_aug
    #     if clip_input_size is None:
    #         clip_input_size = self.clip_input_size
    #
    #     cutouts = []
    #     cutout = T.Resize(clip_input_size, max_size=320)(input)
    #     cutout_h, cutout_w = cutout.shape[-2:]
    #     cutout = self.augs(cutout)
    #     cutouts.append(cutout)
    #     sideY, sideX = input.shape[2:4]
    #     for _ in range(n_aug - 1):
    #         s = (
    #             torch.zeros(
    #                 1,
    #             )
    #             .uniform_(0.6, 1)
    #             .item()
    #         )
    #         h = int(sideY * s)
    #         w = int(sideX * s)
    #         cutout = T.RandomCrop(size=(h, w))(input)
    #         cutout = T.Resize((cutout_h, cutout_w))(cutout)
    #         cutout = self.augs(cutout)
    #         cutouts.append(cutout)
    #
    #     cutouts = torch.cat(cutouts)
    #     return cutouts
    #
    # def get_image_embedding(self, x, aug=True):
    #     if aug:
    #         views = self.augment_input(x)
    #     else:
    #         views = self.basic_transform(x)
    #     if type(views) == list:
    #         image_embeds = []
    #         for view in views:
    #             image_embeds.append(self.encode_image(self.clip_normalize(view)))
    #         image_embeds = torch.cat(image_embeds)
    #     else:
    #         image_embeds = self.encode_image(self.clip_normalize(views))
    #     return image_embeds
    #
    # def encode_image(self, x):
    #     return self.model.encode_image(x)
    #
    # def get_text_embedding(self, text, template, average_embeddings=False):
    #     if type(text) == str:
    #         text = [text]
    #     embeddings = []
    #     for prompt in text:
    #         with torch.no_grad():
    #             embedding = self.model.encode_text(
    #                 clip.tokenize(compose_text_with_templates(prompt, template)).cuda()
    #             )
    #         embeddings.append(embedding)
    #     embeddings = torch.cat(embeddings)
    #     if average_embeddings:
    #         embeddings = embeddings.mean(dim=0, keepdim=True)
    #     return embeddings
    #
    # def get_self_sim(self, x):
    #     x = self.basic_transform(x)
    #     return self.model.calculate_self_sim(x)
    #
    # def calculate_clip_loss(self, outputs, target_embeddings):
    #     # randomly select embeddings
    #     n_embeddings = np.random.randint(1, len(target_embeddings) + 1)
    #     target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n_embeddings,))]
    #
    #     loss = 0.0
    #     for img in outputs:  # avoid memory limitations
    #         img_e = self.get_image_embedding(img.unsqueeze(0))
    #         for target_embedding in target_embeddings:
    #             loss += self.text_criterion(img_e, target_embedding.unsqueeze(0))
    #
    #     # loss /= len(outputs) * len(target_embeddings)
    #     loss /= len(target_embeddings)
    #     return loss

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
