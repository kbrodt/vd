# Video translation

This project is a Minimum Viable Product (MVP) for translating a video with English audio into German. It demonstrates an end-to-end pipeline that includes audio transcription, text translation, speech synthesis with voice cloning, and video-audio synchronization.

## Pipeline Overview

The video translation process is broken down into several sequential steps:

1.  **Audio Extraction**: The original audio is extracted from the input video file.
2.  **Audio Transcription**: The English audio is transcribed into text using OpenAI's Whisper model.
3.  **Text Translation**: The transcribed English text is translated into German using a Helsinki-NLP Opus-MT model.
4.  **Speech Synthesis (Voice Cloning)**: The German text is converted into speech using Coqui's XTTS model. This model clones the voice from the original audio to preserve the speaker's identity.
5.  **Video Synchronization**: The newly generated German audio is merged with the original video frames to create the final translated video.
6.  **Lip Sync**: For a more polished result, a lip-syncing model like Wav2Lip can be integrated to match the speaker's lip movements to the new German audio.

## Prerequisites & Hardware

- GNU/Linux Ubuntu 5.15.0-130-generic
- ffmpeg
- [Python 3.10](https://www.python.org/)
- Nvidia driver version: 565.57.01
- CUDA version: 12.7
- Nvidia GPU with at least 6GB
- [uv](https://docs.astral.sh/uv/)

## Getting started

(Optional) Accept terms for [pyannota speaker diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) model and put your huggingface token in `.env`.

Create virtual environment and run the following script. Don't forget to accept terms if asked.

```bash
uv venv

sh run.sh
```

It takes about 6-9 minutes on the 2018 GPU. The audio will be in `./output/audio/output.wav` and video in `./output/result_voice.mp4`.

## Useful frameworks

### Transcribe audio

- [whisper](https://github.com/openai/whisper)
- [whisperX](https://github.com/m-bain/whisperX)

### Translate text

- [MarianMT](https://huggingface.co/docs/transformers/en/model_doc/marian)

### Generate audio

- [TTS](https://github.com/coqui-ai/TTS)
- [higgs audio v2](https://github.com/boson-ai/higgs-audio)
- [chatterbox](https://github.com/resemble-ai/chatterbox)

### Lip sync

- [Wav2Lip](https://github.com/justinjohn0306/Wav2Lip)
  - [simplified](https://colab.research.google.com/github/justinjohn0306/Wav2Lip/blob/master/Wav2Lip_simplified_v5.ipynb#scrollTo=Qgo-oaI3JU2u)
  - old
- [SadTalker](https://github.com/OpenTalker/SadTalker)
  - old
- [MuseTalk](https://github.com/TMElyralab/MuseTalk)
  - no German

## Assumptions and Limitations

- Single speaker: The current pipeline is optimized for videos with a single, clear speaker. The quality of voice cloning may degrade with multiple speakers or significant background noise.
- Translation accuracy: The translation is handled by a pre-trained machine translation model. While generally accurate, nuances or specific contexts may not be perfectly translated.
- Text-to-speech quality: Preserving personal identity.
- Lip sync quality: Low image resolution. Wav2Lip is trained on images with resolution 96x96.
- Resource Intensive: The transcription and speech synthesis models are computationally expensive.
