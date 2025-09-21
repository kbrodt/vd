#!/usr/bin/env sh


wget -nc -c "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/s3fd.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
wget -nc -c "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" -O 'Wav2Lip/wav2lip_gan.pth'
#wget -nc -c "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth" -O 'Wav2Lip/wav2lip.pth'

uv pip install --no-deps TTS==0.21.0
uv pip install ctranslate2==4.4.0
uv pip install --no-deps whisperx==3.4.2

HF_HUB_ENABLE_HF_TRANSFER=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    uv run \
        --with hf_transfer \
        --with 'protobuf==3.19.6' \
        --with 'packaging==20.9' \
        --with 'pysrt==1.1.2' \
        inference.py \
            --video_url ./data/Tanzania-2.mp4 \
            `#--srt_url ./data/Tanzania-caption.srt` \
            --output_dir ./output \
            --source_language "en" \
            --target_language "de" \
            --use_transcribe \
            --use_translate \
            --use_tts \
            --use_lip_sync \
