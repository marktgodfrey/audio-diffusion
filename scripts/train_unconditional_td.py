# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os
from pathlib import Path
from typing import Optional

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, concatenate_datasets
from diffusers import (
    DanceDiffusionPipeline,
    UNet1DModel,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
import numpy as np
import torch
import torch.nn.functional as F
# from torchvision.transforms import (
#     Compose,
#     Normalize,
#     ToTensor,
# )
from tqdm.auto import tqdm
import librosa


logger = get_logger(__name__)


def stft(sig, n_fft, hop_length, win_length):
    return torch.stft(sig, n_fft, hop_length,
                      win_length=win_length,
                      window=torch.hann_window(win_length, device=sig.device))


def spec(x, n_fft, hop_length, win_length):
    return torch.norm(stft(x, n_fft, hop_length, win_length), p=2, dim=-1)


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def spectral_loss(x_in, x_out, n_fft=2048, hop_length=160, win_length=400):
    spec_in = spec(x_in.squeeze(1), n_fft, hop_length, win_length)
    spec_out = spec(x_out.squeeze(1), n_fft, hop_length, win_length)
    # return torch.mean(norm(spec_in - spec_out))
    return F.smooth_l1_loss(spec_in, spec_out)


def multispectral_loss(x_in, x_out,
                       n_fft=(2048, 1024, 512),
                       hop_length=(160, 80, 40),
                       win_length=(400, 200, 100)):
    losses = []
    assert len(n_fft) == len(hop_length) == len(win_length)
    args = [n_fft, hop_length, win_length]
    for n_fft, hop_length, win_length in zip(*args):
        spec_in = spec(x_in.squeeze(1), n_fft, hop_length, win_length)
        spec_out = spec(x_out.squeeze(1), n_fft, hop_length, win_length)
        losses.append(norm(spec_in - spec_out))
    return torch.mean(sum(losses) / len(losses))


def calculate_bandwidth(dataset, sr=8000, n_fft=2048, hop_length=160, win_length=400, duration=600):
    n_samples = int(sr * duration)
    l1, total, total_sq, n_seen, idx = 0.0, 0.0, 0.0, 0.0, 0
    spec_norm_total, spec_nelem = 0.0, 0.0
    rand_idx = np.random.permutation(len(dataset))
    while n_seen < n_samples:
        samples = dataset[int(rand_idx[idx])]['audio']
        spec_norm = torch.linalg.norm(spec(samples.squeeze(0), n_fft, hop_length, win_length))
        spec_norm_total += spec_norm
        spec_nelem += 1
        n_seen += int(np.prod(samples.shape))
        l1 += torch.sum(torch.abs(samples))
        total += torch.sum(samples)
        total_sq += torch.sum(samples ** 2)
        idx += 1

    mean = total / n_seen
    bandwidth = dict(l2=total_sq / n_seen - mean ** 2,
                     l1=l1 / n_seen,
                     spec=spec_norm_total / spec_nelem)
    return bandwidth


def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    logging_dir = os.path.join(output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.dataset_name is not None:
        if os.path.exists(os.path.join(args.dataset_name, '0')):
            # sharded so let's combine them all
            shard_count = 0
            dataset = None
            while True:
                dataset_path = os.path.join(args.dataset_name, '%d' % shard_count)
                if not os.path.exists(dataset_path):
                    break
                dataset_shard = load_from_disk(dataset_path, args.dataset_config_name)["train"]
                if dataset is None:
                    dataset = dataset_shard
                else:
                    dataset = concatenate_datasets([dataset, dataset_shard])
                shard_count += 1
            dataset = dataset.with_format("torch")
        elif os.path.exists(args.dataset_name):
            dataset = load_from_disk(args.dataset_name, args.dataset_config_name)["train"].with_format("torch")
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )

    print('loaded dataset with %d samples' % len(dataset))

    # bandwidth = calculate_bandwidth(dataset, sr=args.sample_rate)
    # bandwidth['spec'] = 0.001 * bandwidth['spec']  # hacks
    # print('bandwidth: %s', bandwidth)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.from_pretrained is not None:
        pipeline = DanceDiffusionPipeline.from_pretrained(args.from_pretrained)
        model = pipeline.unet
        noise_scheduler = pipeline.scheduler
    else:
        # model = UNet1DModel(
        #     sample_size=win_length,
        #     in_channels=1,
        #     out_channels=1,
        #     layers_per_block=2,
        #     block_out_channels=(128, 128, 256, 256, 512, 512),
        #     down_block_types=(
        #         "DownBlock1D",
        #         "DownBlock1D",
        #         "DownBlock1D",
        #         "DownBlock1D",
        #         "AttnDownBlock1D",
        #         "DownBlock1D",
        #     ),
        #     up_block_types=(
        #         "UpBlock1D",
        #         "AttnUpBlock1D",
        #         "UpBlock1D",
        #         "UpBlock1D",
        #         "UpBlock1D",
        #         "UpBlock1D",
        #     ),
        # )

        # model = UNet1DModel(
        #     sample_size=args.win_length,
        #     block_out_channels=(128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512),
        #     extra_in_channels=16,
        #     sample_rate=args.sample_rate,
        #     in_channels=1,
        #     out_channels=1,
        #     flip_sin_to_cos=True,
        #     use_timestep_embedding=False,
        #     time_embedding_type="fourier",
        #     mid_block_type="UNetMidBlock1D",
        #     down_block_types=("DownBlock1DNoSkip",
        #                       "DownBlock1D",
        #                       "DownBlock1D",
        #                       "DownBlock1D",
        #                       "DownBlock1D",
        #                       "DownBlock1D",
        #                       "DownBlock1D",
        #                       "DownBlock1D",
        #                       "AttnDownBlock1D",
        #                       "AttnDownBlock1D",
        #                       "AttnDownBlock1D",
        #                       "AttnDownBlock1D",
        #                       "AttnDownBlock1D"),
        #     up_block_types=("AttnUpBlock1D",
        #                     "AttnUpBlock1D",
        #                     "AttnUpBlock1D",
        #                     "AttnUpBlock1D",
        #                     "AttnUpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1D",
        #                     "UpBlock1DNoSkip"),
        # )

        model = UNet1DModel(
            sample_size=args.win_length,
            block_out_channels=(128, 256, 512, 512, 512, 512, 512),
            extra_in_channels=16,
            sample_rate=args.sample_rate,
            in_channels=1,
            out_channels=1,
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            time_embedding_type="fourier",
            mid_block_type="UNetMidBlock1D",
            down_block_types=("DownBlock1DNoSkip",
                              "DownBlock1D",
                              "DownBlock1D",
                              "DownBlock1D",
                              "AttnDownBlock1D",
                              "AttnDownBlock1D",
                              "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D",
                            "AttnUpBlock1D",
                            "AttnUpBlock1D",
                            "UpBlock1D",
                            "UpBlock1D",
                            "UpBlock1D",
                            "UpBlock1DNoSkip"),
        )

        if args.scheduler == "ddpm":
            noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_steps)
        elif args.scheduler == "ddim":
            noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps)
        else:
            noise_scheduler = PNDMScheduler(num_train_timesteps=args.num_train_steps, skip_prk_steps=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(output_dir).name,
                                           token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        repo = Repository(output_dir, clone_from=repo_name)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    # if accelerator.is_main_process:
    #     with torch.no_grad():
    #         print('debug start!')
    #         generator = torch.Generator(device=model.device).manual_seed(42)
    #         scheduler = DDIMScheduler()
    #         scheduler.set_timesteps(50)save_images_epochs
    #         init = torch.randn((args.eval_batch_size, 1, win_length),
    #                            generator=generator,
    #                            device=model.device)
    #         audios = init
    #         unwrapped_model = accelerator.unwrap_model(model)
    #         for t in scheduler.timesteps:
    #             print(audios.size())
    #             noise_pred = unwrapped_model(audios, t)["sample"]
    #             audios = scheduler.step(model_output=noise_pred, timestep=t, sample=audios,
    #                                     generator=generator, eta=0.)["prev_sample"]
    #         print('debug end!')

    torch.autograd.set_detect_anomaly(True)

    global_step = 0
    for epoch in range(args.num_epochs):

        torch.cuda.empty_cache()

        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch['audio']
            clean_images = clean_images[:, :, :args.win_length]

            for _ in range(args.num_noise_loops):
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz, ),
                    device=clean_images.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images,
                                                         noise,
                                                         timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    output = model(noisy_images, timesteps)
                    noise_pred = output["sample"]
                    td_loss = F.mse_loss(noise_pred, noise)
                    # td_loss = F.smooth_l1_loss(noise_pred, noise)
                    # loss = F.l1_loss(noise_pred, noise)
                    # multispec_loss = multispectral_loss(noise_pred, noise)
                    # spec_loss = spectral_loss(noise_pred, noise)

                    loss = td_loss
                    # loss = td_loss / bandwidth['l2'] + multispec_loss / bandwidth['spec']
                    # loss = td_loss / bandwidth['l2'] + spec_loss / bandwidth['spec']

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    if args.use_ema:
                        ema_model.step(model)
                    optimizer.zero_grad()

                global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "td_loss": td_loss.detach().item(), #(td_loss / bandwidth['l2']).detach().item(),
                    #"spec_loss": (spec_loss / bandwidth['spec']).detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                if args.use_ema:
                    logs["ema_decay"] = ema_model.decay
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            progress_bar.update(1)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline = DanceDiffusionPipeline(
                    unet=accelerator.unwrap_model(
                        ema_model.averaged_model if args.use_ema else model),
                    scheduler=noise_scheduler,
                )
                pipeline.save_pretrained(output_dir)

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}",
                                     blocking=False,
                                     auto_lfs_prune=True)

            if (epoch + 1) % args.save_images_epochs == 0:
                with torch.no_grad():
                    generator = torch.Generator(device=clean_images.device).manual_seed(42)
                    scheduler = PNDMScheduler(num_train_timesteps=args.num_train_steps, skip_prk_steps=False)
                    scheduler.set_timesteps(args.num_train_steps)
                    init = torch.randn((args.eval_batch_size, 1, args.win_length),
                                       generator=generator,
                                       device=clean_images.device)
                    audios = init
                    unwrapped_model = accelerator.unwrap_model(model)
                    for t in scheduler.timesteps:
                        noise_pred = unwrapped_model(audios, t)["sample"]
                        audios = scheduler.step(model_output=noise_pred, timestep=t, sample=audios)["prev_sample"]
                    audios = audios.clamp(-1., 1.).detach().cpu().numpy()
                    for i, audio in enumerate(audios):
                        accelerator.trackers[0].writer.add_audio(
                            f"test_audio_{i}",
                            # normalize(audio, axis=1),
                            audio,
                            epoch,
                            sample_rate=args.sample_rate,
                        )

    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_noise_loops", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=False)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--win_length", type=int, default=32768)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddpm",
                        help="ddpm or ddim or pndm")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
