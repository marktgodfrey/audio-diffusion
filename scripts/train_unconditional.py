# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os
from pathlib import Path
from typing import Optional

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, concatenate_datasets
from diffusers import (
    AudioDiffusionPipeline,
    DDPMScheduler,
    UNet2DModel,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers.pipelines.audio_diffusion import Mel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
# import numpy as np
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

    # Determine image resolution
    # resolution = dataset[0]["image"].height, dataset[0]["image"].width
    resolution = dataset[0]["melspec"].shape

    # augmentations = Compose([
    #     ToTensor(),
    #     Normalize([0.5], [0.5]),
    # ])
    #
    # def transforms(examples):
    #     if args.vae is not None and vqvae.config["in_channels"] == 3:
    #         images = [
    #             augmentations(image.convert("RGB"))
    #             for image in examples["image"]
    #         ]
    #     else:
    #         images = [augmentations(image) for image in examples["image"]]
    #     return {"input": images}
    #
    # dataset.set_transform(transforms)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    vqvae = None
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae)
        except EnvironmentError:
            vqvae = AudioDiffusionPipeline.from_pretrained(args.vae).vqvae
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = (vqvae.encode(
                torch.zeros((1, 1) +
                            resolution)).latent_dist.sample().shape[2:])

    if args.from_pretrained is not None:
        pipeline = AudioDiffusionPipeline.from_pretrained(args.from_pretrained)
        # mel = pipeline.mel
        model = pipeline.unet
        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae
    else:
        model = UNet2DModel(
            sample_size=resolution if vqvae is None else latent_resolution,
            in_channels=1
            if vqvae is None else vqvae.config["latent_channels"],
            out_channels=1
            if vqvae is None else vqvae.config["latent_channels"],
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps)

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

    mel = Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

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
            # clean_images = batch["input"]
            clean_images = (batch['melspec'] - 0.5) * 2.
            clean_images = clean_images.unsqueeze(1)

            # if accelerator.is_main_process:
            #     accelerator.trackers[0].writer.add_images(
            #         "train_samples", (clean_images / 2. + 0.5), global_step)

            if vqvae is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(
                        clean_images).latent_dist.sample()
                # Scale latent images to ensure approximately unit variance
                clean_images = clean_images * 0.18215

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
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                # if accelerator.is_main_process:
                #     images = noise_scheduler.step(model_output=noise_pred, timestep=timesteps[-1], sample=noisy_images)["prev_sample"]
                #     images = (images / 2. + 0.5).clamp(0., 1.).detach().cpu().numpy()
                #     print(images.shape)
                #     accelerator.trackers[0].writer.add_images(
                #         "train_output", images, global_step)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

                # if accelerator.is_main_process:
                #     with torch.no_grad():
                #         generator = torch.Generator(device=clean_images.device)
                #         scheduler = DDIMScheduler()
                #         scheduler.set_timesteps(50)
                #         init = torch.randn((4, 1, resolution[0], resolution[1]),
                #                            generator=generator,
                #                            device=clean_images.device)
                #         images = init
                #         unwrapped_model = accelerator.unwrap_model(model)
                #         for t in scheduler.timesteps:
                #             noise_pred = unwrapped_model(images, t)["sample"]
                #             images = scheduler.step(model_output=noise_pred, timestep=t, sample=images,
                #                                     generator=generator, eta=0.)["prev_sample"]
                #         images = (images / 2. + 0.5).clamp(0., 1.).detach().cpu().numpy()
                #         print(images.shape)
                #         accelerator.trackers[0].writer.add_images(
                #             "train_output", images, global_step)

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline = AudioDiffusionPipeline(
                    vqvae=vqvae,
                    unet=accelerator.unwrap_model(
                        ema_model.averaged_model if args.use_ema else model),
                    mel=mel,
                    scheduler=noise_scheduler,
                )

                pipeline.save_pretrained(output_dir)

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}",
                                     blocking=False,
                                     auto_lfs_prune=True)

            if (epoch + 1) % args.save_images_epochs == 0:
                # generator = torch.Generator(
                #     device=clean_images.device).manual_seed(42)
                # # run pipeline in inference (sample random noise and denoise)
                # images, (sample_rate,
                #          audios) = pipeline(generator=generator,
                #                             batch_size=args.eval_batch_size,
                #                             return_dict=False)
                #
                # # denormalize the images and save to tensorboard
                # images = np.array([
                #     np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                #         (len(image.getbands()), image.height, image.width))
                #     for image in images
                # ])
                # accelerator.trackers[0].writer.add_images(
                #     "test_samples", images, epoch)
                # for _, audio in enumerate(audios):
                #     accelerator.trackers[0].writer.add_audio(
                #         f"test_audio_{_}",
                #         normalize(audio),
                #         epoch,
                #         sample_rate=sample_rate,
                #     )

                with torch.no_grad():
                    generator = torch.Generator(device=clean_images.device).manual_seed(42)
                    scheduler = DDIMScheduler()
                    scheduler.set_timesteps(50)
                    init = torch.randn((args.eval_batch_size, 1, resolution[0], resolution[1]),
                                       generator=generator,
                                       device=clean_images.device)
                    images = init
                    unwrapped_model = accelerator.unwrap_model(model)
                    for t in scheduler.timesteps:
                        noise_pred = unwrapped_model(images, t)["sample"]
                        images = scheduler.step(model_output=noise_pred, timestep=t, sample=images,
                                                generator=generator, eta=0.)["prev_sample"]
                    images = (images / 2. + 0.5).clamp(0., 1.).detach().cpu().numpy()
                    accelerator.trackers[0].writer.add_images(
                        "test_samples", images, global_step)
                    for i, image in enumerate(images):
                        log_S = librosa.db_to_power(image * mel.top_db - mel.top_db)
                        audio = librosa.feature.inverse.mel_to_audio(log_S,
                                                                     sr=mel.sr,
                                                                     n_fft=mel.n_fft,
                                                                     hop_length=mel.hop_length,
                                                                     n_iter=mel.n_iter)
                        accelerator.trackers[0].writer.add_audio(
                            f"test_audio_{i}",
                            normalize(audio, axis=1),
                            epoch,
                            sample_rate=mel.sample_rate,
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
    parser.add_argument("--use_ema", type=bool, default=True)
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
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddpm",
                        help="ddpm or ddim")
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
