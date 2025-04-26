#!/bin/bash

rm -rf mkdir assets/demo1_video
mkdir assets/demo1_video
ffmpeg -i assets/demo1_video.mp4 assets/demo1_video/output_%04d.png

exec python -m scripts.inference \
    --unet_config_path "configs/unet/stage2.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 2.0 \
    --video_path "assets/demo1_video" \
    --audio_path "assets/demo1_audio.wav" \
    --frames_out_path "video_out"
