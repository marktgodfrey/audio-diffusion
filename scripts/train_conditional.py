# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py

import argparse
import os
import random
from pathlib import Path
from typing import Optional

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, concatenate_datasets
from diffusers import (
    AudioDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers.pipelines.audio_diffusion import Mel
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

from gomin.models import GomiGAN
from gomin.config import GANConfig


logger = get_logger(__name__)

MAX_MIN_LOSSES = 3


def mel(x, n_fft, hop_length, n_mels, sr, top_db):
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft).to(x.device),
        return_complex=True,
        center=True,
    )
    magnitude = torch.square(torch.abs(stft))
    nf = magnitude.shape[2]
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=2 * (nf - 1),
        n_mels=n_mels
    )
    mel_basis = torch.from_numpy(mel_basis).to(x.device)

    spec = magnitude.transpose(2, -1) @ mel_basis.T
    spec = spec.transpose(-1, 2)

    spec = torch.clamp(spec, min=1e-10)
    spec_db = 10.0 * torch.log10(spec)
    spec_db -= 10.0 * torch.log10(torch.max(spec))
    spec_db = torch.maximum(spec_db, spec_db.max() - top_db)
    spec_db_norm = ((spec_db + top_db) / top_db).clip(0., 1.)
    return spec_db_norm


def multimel_loss(x_in, x_out,
                  n_mels=(5, 10, 20, 40, 80, 160, 320),
                  hop_lengths=(8, 16, 32, 64, 128, 256, 1024),
                  win_lengths=(32, 64, 128, 256, 512, 1024, 2048),
                  sr=24000,
                  top_db=80):
    losses = []
    assert len(n_mels) == len(hop_lengths) == len(win_lengths)
    args = [n_mels, hop_lengths, win_lengths]
    for n_mel, hop_length, win_length in zip(*args):
        spec_in = mel(x_in.squeeze(1), win_length, hop_length, n_mel, sr, top_db)
        spec_out = mel(x_out.squeeze(1), win_length, hop_length, n_mel, sr, top_db)
        losses.append(F.l1_loss(spec_in, spec_out))
    return torch.mean(sum(losses) / len(losses))


def encoding_from_batch(metadata, labels):
    dim = len(labels)
    return torch.tensor([[1. if a == labels[i] else 0. for a in metadata]
                         for i in range(dim)]).transpose(0, 1).view(-1, 1, dim)


def encoding_for_sampling(batch_size, dim=1):
    if dim == 1:
        return torch.tensor(int(args.eval_batch_size/2)*[0.] + int(args.eval_batch_size/2)*[1.]).view(-1, 1, 1)
    if dim > 1:
        return torch.eye(dim).repeat(int(batch_size/dim)+1, 1)[:batch_size].view(-1, 1, dim)
    return None


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
    resolution = dataset[0]["spec"].shape
    # print(resolution)

    encoding_dim = len(args.cond_labels)
    if args.uncond:
        print('unconditional training!')
    else:
        print('labels for conditioning: %s' % args.cond_labels)

    vocoder = GomiGAN.from_pretrained(pretrained_model_path="/workspace/code/audio-diffusion/gomin/models/gan_state_dict.pt", **GANConfig().__dict__).to('cuda')
    for p in vocoder.parameters():
        p.requires_grad = False

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
        model = UNet2DConditionModel(
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
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=encoding_dim,
        )

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps,
            beta_schedule=args.beta_schedule)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps,
            beta_schedule=args.beta_schedule)
    noise_scheduler.set_timesteps(args.num_train_steps)

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
        decay=args.ema_max_decay,
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
    min_losses = []
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
            clean_images = (batch['spec'] - 0.5) * 2.
            clean_images = clean_images.unsqueeze(1)

            if args.uncond:
                encoding = torch.zeros(clean_images.size(0), 1, encoding_dim).to(clean_images.device)
            else:
                encoding = encoding_from_batch(batch[args.cond_key], args.cond_labels).to(clean_images.device)
                if args.cond_dropout > 0.:
                    d = torch.bernoulli(torch.tensor([(1. - args.cond_dropout)] * clean_images.size(0)))
                    d = d.repeat_interleave(encoding_dim).view(-1, 1, encoding_dim).to(clean_images.device)
                    encoding *= d

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
                noise_scheduler.config.num_train_timesteps,
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
                output = model(noisy_images, timesteps, encoding)
                noise_pred = output["sample"]

                # loss = F.mse_loss(noise_pred, noise)
                loss = F.smooth_l1_loss(noise_pred, noise)
                # loss = F.l1_loss(noise_pred, noise)

                # in_audio = batch['audio']
                # in_audio = vocoder(noisy_images)
                # out_images = []
                # for i, t in enumerate(timesteps):
                #     out_image = noise_scheduler.step(model_output=noise_pred[i],
                #                                      timestep=t,
                #                                      sample=noisy_images[i])["prev_sample"]
                #     out_image = (out_image / 2. + 0.5).clamp(0., 1.)
                #     out_images.append(out_image)
                # out_images = torch.cat(out_images)
                # out_audio = vocoder(out_images)
                #
                # loss = multimel_loss(in_audio, out_audio)

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

            if accelerator.is_main_process:
                this_loss = loss.detach().item()
                should_save = False
                if len(min_losses) < MAX_MIN_LOSSES:
                    min_losses.append((this_loss, global_step))
                    min_losses = sorted(min_losses, key=lambda l: l[0])
                else:
                    for min_loss_idx, min_loss in enumerate(min_losses):
                        if this_loss < min_loss[0]:
                            min_losses.pop(min_loss_idx)
                            min_losses.insert(min_loss_idx, (this_loss, global_step))
                            should_save = True
                            break

                if epoch > 10 and should_save:
                    pipeline = AudioDiffusionPipeline(
                        vqvae=vqvae,
                        unet=accelerator.unwrap_model(
                            ema_model.averaged_model if args.use_ema else model),
                        mel=mel,
                        scheduler=noise_scheduler,
                    )
                    pipeline.save_pretrained(os.path.join(output_dir, str(global_step)))

                    with torch.no_grad():
                        generator = torch.Generator(device=clean_images.device).manual_seed(42)
                        scheduler = DDIMScheduler(beta_schedule=args.beta_schedule)
                        scheduler.set_timesteps(50)
                        init = torch.randn((args.eval_batch_size, 1, resolution[0], resolution[1]),
                                           generator=generator,
                                           device=clean_images.device)
                        if args.uncond:
                            encoding = torch.zeros(args.eval_batch_size, 1, encoding_dim).to(clean_images.device)
                        else:
                            encoding = encoding_for_sampling(args.eval_batch_size, dim=encoding_dim).to(clean_images.device)
                        images = init
                        unwrapped_model = accelerator.unwrap_model(model)
                        for t in scheduler.timesteps:
                            noise_pred = unwrapped_model(images, t, encoding)["sample"]
                            images = scheduler.step(model_output=noise_pred, timestep=t, sample=images,
                                                    generator=generator, eta=0.)["prev_sample"]
                        images = (images / 2. + 0.5).clamp(0., 1.).detach()
                        for i, image in enumerate(images):
                            # mag = librosa.db_to_power(image.cpu().numpy() * mel.top_db - mel.top_db)
                            # # linear specgram, add back DC
                            # mag_pad = np.zeros(((mel.y_res + 1), mel.x_res))
                            # mag_pad[1:(mel.y_res + 1), :] = np.power(mag, 0.5)
                            # audio = librosa.griffinlim(mag_pad,
                            #                            n_fft=(mel.y_res * 2),
                            #                            win_length=mel.n_fft,
                            #                            hop_length=mel.hop_length)
                            audio = vocoder(image).detach().squeeze().cpu().numpy()
                            if args.uncond:
                                accelerator.trackers[0].writer.add_audio(
                                    f"best_audio_{i}",
                                    normalize(audio),
                                    global_step,
                                    sample_rate=args.sample_rate
                                )
                            else:
                                accelerator.trackers[0].writer.add_audio(
                                    f"best_audio_{i}_{args.cond_labels[i % len(args.cond_labels)]}",
                                    normalize(audio),
                                    global_step,
                                    sample_rate=args.sample_rate
                                )

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:

            print(min_losses)

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
                    scheduler = DDIMScheduler(beta_schedule=args.beta_schedule)
                    scheduler.set_timesteps(50)
                    init = torch.randn((args.eval_batch_size, 1, resolution[0], resolution[1]),
                                       generator=generator,
                                       device=clean_images.device)
                    if args.uncond:
                        encoding = torch.zeros(args.eval_batch_size, 1, encoding_dim).to(clean_images.device)
                    else:
                        random.seed(42)
                        encoding = encoding_for_sampling(args.eval_batch_size, dim=encoding_dim).to(clean_images.device)
                    images = init
                    unwrapped_model = accelerator.unwrap_model(model)
                    for t in scheduler.timesteps:
                        noise_pred = unwrapped_model(images, t, encoding)["sample"]
                        images = scheduler.step(model_output=noise_pred, timestep=t, sample=images,
                                                generator=generator, eta=0.)["prev_sample"]
                    images = (images / 2. + 0.5).clamp(0., 1.).detach()
                    accelerator.trackers[0].writer.add_images(
                        "test_samples", images.cpu().numpy(), global_step)
                    for i, image in enumerate(images):
                        # mag = librosa.db_to_power(image * mel.top_db - mel.top_db)

                        # audio = librosa.feature.inverse.mel_to_audio(mag,
                        #                                              sr=mel.sr,
                        #                                              n_fft=mel.n_fft,
                        #                                              hop_length=mel.hop_length,
                        #                                              n_iter=mel.n_iter)

                        # # linear specgram, add back DC
                        # mag_pad = np.zeros(((mel.y_res + 1), mel.x_res))
                        # mag_pad[1:(mel.y_res + 1), :] = np.power(mag, 0.5)
                        # audio = librosa.griffinlim(mag_pad,
                        #                            n_fft=(mel.y_res * 2),
                        #                            win_length=mel.n_fft,
                        #                            hop_length=mel.hop_length)

                        audio = vocoder(image).detach().squeeze().cpu().numpy()
                        if args.uncond:
                            accelerator.trackers[0].writer.add_audio(
                                f"test_audio_{i}",
                                normalize(audio),
                                epoch,
                                sample_rate=args.sample_rate,
                            )
                        else:
                            accelerator.trackers[0].writer.add_audio(
                                f"test_audio_{i}_{args.cond_labels[i % len(args.cond_labels)]}",
                                normalize(audio),
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
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler", type=str, default="ddpm", help="ddpm or ddim")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="linear, scaled_linear, or squaredcos_cap_v2")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )
    parser.add_argument("--cond_key", type=str, help="key of metadata dict for conditioning")
    parser.add_argument("--cond_labels", nargs='+', help="list of labels for conditioning")
    parser.add_argument("--cond_dropout", type=float, default=0.0, help="conditional drop-out probability")
    parser.add_argument("--uncond", action='store_true', help="override conditioning (for CFG fine-tuning)")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
