import os
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import wandb
import datetime
import argparse

from config import *
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM_eval


def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0
    return img


def wandb_init(config):
    wandb.init(
        project="dreamdiffusion",
        group='eval',
        anonymous="allow",
        config=config,
        reinit=True,
    )


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


def get_args_parser():
    parser = argparse.ArgumentParser('Parameterized DreamDiffusion EEG Evaluation')
    parser.add_argument('--root', type=str, default='../dreamdiffusion/')
    parser.add_argument('--dataset', type=str, default='GOD')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--subject', type=int, default=4,
                        help='EEG subject id. Use 4 for the original single-subject setup.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of generated candidates per EEG sample. Defaults to checkpoint config.num_samples.')
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='Number of diffusion sampling steps. Defaults to checkpoint config.ddim_steps.')
    parser.add_argument('--train_limit', type=int, default=10,
                        help='Number of train preview EEG samples to generate. Set 0 to skip train preview.')
    parser.add_argument('--test_limit', '--limit', dest='test_limit', type=int, default=None,
                        help='Number of test EEG samples to generate. Omit for all test samples.')
    parser.add_argument('--skip_train_preview', action='store_true',
                        help='Skip the train preview generation and only generate test samples.')

    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to dataset splits.')
    parser.add_argument('--eeg_signals_path', type=str, default=None,
                        help='Path to EEG signals data.')
    parser.add_argument('--config_patch', type=str, default=None,
                        help='sd config path.')
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='imagenet path.')
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    root = args.root

    sd = torch.load(args.model_path, map_location='cpu')
    config = sd['config']
    config.root_path = root

    num_samples = args.num_samples if args.num_samples is not None else config.num_samples
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else config.ddim_steps
    train_limit = 0 if args.skip_train_preview else args.train_limit
    test_limit = args.test_limit

    if num_samples <= 0:
        raise ValueError('--num_samples must be positive.')
    if ddim_steps <= 0:
        raise ValueError('--ddim_steps must be positive.')
    if train_limit is not None and train_limit < 0:
        raise ValueError('--train_limit must be >= 0.')
    if test_limit is not None and test_limit < 0:
        raise ValueError('--test_limit/--limit must be >= 0.')

    output_path = os.path.join(
        config.root_path,
        'results',
        'eval',
        '%s' % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crop_pix = int(config.crop_ratio * config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        # random_crop(config.img_size-crop_pix, p=0.5),
        # transforms.Resize((256, 256)),
        channel_last,
    ])
    img_transform_test = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        channel_last,
    ])

    dataset_train, dataset_test = create_EEG_dataset(
        eeg_signals_path=args.eeg_signals_path,
        splits_path=args.splits_path,
        imagenet_path=args.imagenet_path,
        image_transform=[img_transform_train, img_transform_test],
        subject=args.subject,
    )
    num_voxels = dataset_test.dataset.data_len

    generative_model = eLDM_eval(
        args.config_patch,
        num_voxels,
        device=device,
        pretrain_root=config.pretrain_gm_path,
        logger=config.logger,
        ddim_steps=ddim_steps,
        global_pool=config.global_pool,
        use_time_cond=config.use_time_cond,
    )
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')
    print(f'output_path: {output_path}')
    print(f'subject: {args.subject}')
    print(f'num_samples per EEG sample: {num_samples}')
    print(f'ddim_steps: {ddim_steps}')
    print(f'train_limit: {train_limit}')
    print(f'test_limit: {test_limit}')

    state = sd['state']
    os.makedirs(output_path, exist_ok=True)

    if train_limit:
        grid, _ = generative_model.generate(
            dataset_train,
            num_samples,
            ddim_steps,
            config.HW,
            train_limit,
        )
        grid_imgs = Image.fromarray(grid.astype(np.uint8))
        grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))

    grid, samples = generative_model.generate(
        dataset_test,
        num_samples,
        ddim_steps,
        config.HW,
        limit=test_limit,
        state=state,
        output_path=output_path,
    )
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))
