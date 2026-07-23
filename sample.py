import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import random
import numpy as np
import torch
from torchvision import transforms
from diffusion.diffusion import CLIP_Semantic_extractor
from text2live_util.clip_extractor import ClipExtractor
from text2live_util.util import get_augmentations_template

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from common_utils.image import imread
from common_utils.common import two_tuple
from common_utils.resize_right import resize
from common_utils.video import torchvid2mp4
from config import *
from diffusion.conditional_diffusion import ConditionalDiffusion
from diffusion.diffusion import Diffusion
from diffusion.diffusion_utils import save_diffusion_sample
from models.nextnet import NextNet, ModifiedNextNet
from tool.resizer import Resizer
from einops import rearrange


def get_model_path(image_name, version_name):
    return os.path.join('lightning_logs', image_name, version_name, 'checkpoints', 'last.ckpt')


def create_sample_directory(cfg, extra_path=''):
    if cfg.sample_count == 1:
        sample_directory = os.path.join(cfg.output_dir, os.path.split(cfg.image_name)[0], cfg.run_name, extra_path)
    else:
        sample_directory = os.path.join(cfg.output_dir, cfg.image_name, cfg.run_name, extra_path)
    os.makedirs(sample_directory, exist_ok=True)
    print(f'Sample directory: {sample_directory}')
    return sample_directory


def noise_img(img, model, t):
    """
    Add noise (equivalent to t steps of a forward diffusion process) to an image.

    Args:
        img (torch.Tensor): Image to add noise to.
        model (Diffusion or ConditionalDiffusion): Diffusion model with "q_sample" implementation.
        t (int): Number of forward diffusion steps to perform.
    """
    batch_size = img.shape[0]
    if isinstance(model, Diffusion):
        noisy_img = model.q_sample(img, t)
    elif isinstance(model, ConditionalDiffusion):
        continuous_sqrt_alpha_hat = torch.FloatTensor(
            np.random.uniform(model.sqrt_alphas_hat_prev[t - 1], model.sqrt_alphas_hat_prev[t], size=batch_size)).to(
            img.device).view(batch_size, -1)
        noisy_img = model.q_sample(img, continuous_sqrt_alpha_hat.view(-1, 1, 1, 1))
    else:
        raise Exception

    return noisy_img


def generate_diverse_samples(cfg):
    """
    Generates diverse image samples from a single image DDPM trained model.

    Args:
        cfg (Config):
            Configuration object.
    """
    # Create sample directory
    sample_directory = create_sample_directory(cfg)

    # Load model
    path = get_model_path(cfg.image_name, cfg.run_name)
    model = Diffusion.load_from_checkpoint(path, model=ModifiedNextNet(depth=cfg.network_depth,
                                                                       use_cbam=cfg.use_cbam, use_concat=cfg.use_concat,
                                                                       use_semantic=cfg.use_semantic,
                                                                       ),
                                           timesteps=cfg.diffusion_timesteps,
                                           training_target='x0',
                                           use_semantic=cfg.use_semantic,
                                           noise_schedule='linear').cuda()

    ts = 0.98 * cfg.diffusion_timesteps

    image = imread(f'./images/{cfg.image_name}')

    if cfg.use_semantic:
        assert image is not None
        transform_list = [
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ]
        transform = transforms.Compose(transform_list)
        image = transform(image).cuda()
        model_ex = CLIP_Semantic_extractor().cuda()
        img_semantic = model_ex(image)
    else:
        image = (image * 2 - 1).cuda()
        img_semantic = None

    # img_semantic_npy = rearrange(img_semantic[0], 'c h w -> (h w) c').contiguous().cpu().numpy()
    # dir_path = './tsne/img_semantic/'
    # os.makedirs(dir_path, exist_ok=True)
    # np.save(f'{dir_path}/img_semantic.npy', img_semantic_npy)

    if cfg.task == 'edit' or cfg.task == 'paint' or cfg.task == 'style_trans':
        image = (imread(f'./images/{cfg.task}/{cfg.dist_image_name}') * 2 - 1).cuda()
        ts = 0

    if cfg.sample_size is None:
        size = tuple(image.shape[-2:])
    else:
        size = two_tuple(cfg.sample_size)

    if cfg.text_guide:
        assert cfg.text_input is not None, 'text_input is None'
        t2l_clip_extractor = ClipExtractor()
        text_embedds = t2l_clip_extractor.get_text_embedding(cfg.text_input, template=get_augmentations_template('hr'))
    else:
        t2l_clip_extractor = None
        text_embedds = None

    # Sample and save images
    batch_size = 1
    samples = []

    # load domain-independent feature extractor
    N = 32
    shape = (batch_size, 3, image.shape[-2], image.shape[-1])
    shape_d = (batch_size, 3, int(image.shape[-2] / N), int(image.shape[-1] / N))
    down = Resizer(shape, 1 / N).cuda()
    up = Resizer(shape_d, N).cuda()
    die = (down, up)

    for i in tqdm(range(0, cfg.sample_count, batch_size)):
        samples.append(model.modified_sample(image_size=size, batch_size=min(batch_size, cfg.sample_count - i),
                                             img_semantic=img_semantic,
                                             source_image=image,
                                             die=die,
                                             t2l_clip_extractor=t2l_clip_extractor,
                                             text_embedds=text_embedds,
                                             ts=ts,
                                             ))

    samples = torch.cat(samples, dim=0)
    # save_diffusion_sample(samples, os.path.join(sample_directory, 'sample.jpg'))
    save_diffusion_sample(samples, os.path.join(sample_directory, os.path.split(cfg.image_name)[1]))


def main():
    cfg = Config()
    cfg = parse_cmdline_args_to_config(cfg)

    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.available_gpus

    log_config(cfg)

    generate_diverse_samples(cfg)


if __name__ == '__main__':
    main()
