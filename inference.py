import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import subprocess
import textwrap
from pathlib import Path

import nltk
import pysrt
import torch
torch.set_num_threads(1)
import whisperx
from TTS.api import TTS
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from nltk.tokenize import sent_tokenize
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    set_seed,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Choose video URL')

    parser.add_argument('--video_url', type=str, help='Single video URL', required=True)
    parser.add_argument('--srt_url', type=str, help='Single srt URL', default=None, required=False)
    parser.add_argument('--output_dir', type=str, help='Output directory', default="./output", required=False)

    parser.add_argument('--source_language', type=str, help='Video source language', default="en", required=False)
    parser.add_argument('--target_language', type=str, help='Video target language', default="de", required=False)

    parser.add_argument('--whisper_model', type=str, help='Chose the whisper model based on your device requirements', default="medium")
    parser.add_argument('--diarization_model_name', type=str, help='Chose the diarization model based on your device requirements', default="pyannote/speaker-diarization-3.1")
    parser.add_argument('--tts_model_name', type=str, help='Chose the TTS model based on your device requirements', default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument('--lip_sync_model_name', type=str, help='Chose the Lip Sync model based on your device requirements', default="wav2lip_gan.pth")

    parser.add_argument('--seed', type=int, help='Chose the whisper model based on your device requirements', default=314159)

    parser.add_argument('--use_transcribe', action="store_true", help='Extract text from audio')
    parser.add_argument('--use_translate', action="store_true", help='Translate the extraced text')
    parser.add_argument('--use_tts', action="store_true", help='Generate speech')
    parser.add_argument('--use_lip_sync', action="store_true", help='Lip synchronization of the resut audio to the synthesized video')

    args = parser.parse_args(args)

    return args


class VideoDubbing:
    def __init__(
        self,
        video_path,
        srt_path,
        output_dir,
        source_language="en",
        target_language="de",
        use_transcribe=True,
        use_translate=True,
        use_tts=True,
        use_lip_sync=True,
        huggingface_auth_token=None,
        whisper_model="medium",
        diarization_model_name="pyannote/speaker-diarization-3.1",
        tts_model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        lip_sync_model_name="wav2lip_gan.pth",
    ):
        self.video_path = video_path
        self.srt_path = srt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.speakers_audio_dir = self.output_dir / "speakers_audio"
        self.speakers_audio_dir.mkdir(parents=True, exist_ok=True)

        self.source_language = source_language
        self.target_language = target_language

        self.use_transcribe = use_transcribe
        self.use_translate = use_translate
        self.use_tts = use_tts
        self.use_lip_sync = use_lip_sync

        self.whisper_model = whisper_model
        self.lip_sync_model_name = lip_sync_model_name
        self.huggingface_auth_token = huggingface_auth_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.huggingface_auth_token:
            self.diarization_pipeline = Pipeline.from_pretrained(
                diarization_model_name,
                use_auth_token=self.huggingface_auth_token,
            ).to(self.device)

        if self.use_translate:
            self.mt_model_name = f"Helsinki-NLP/opus-mt-{self.source_language}-{self.target_language}"
            self.mt_tokenizer = MarianTokenizer.from_pretrained(self.mt_model_name)
            self.mt_model = MarianMTModel.from_pretrained(self.mt_model_name).eval().to(self.device)

        if self.use_tts:
            self.tts_model = TTS(tts_model_name).eval().to(self.device)

    def get_time_stamped_setnteces(self, segments, use_whisper_fast=False):
        time_stamped = []
        full_text = []
        for segment in segments:
            for word in segment["words"]:
                w = word["word"]
                if not use_whisper_fast:
                    w = " " + w

                time_stamped.append([w, word["start"], word["end"]])
                full_text.append(w)
                print(time_stamped[-1])

        full_text = "".join(full_text)

        sentences = sent_tokenize(full_text)
        print(sentences)

        time_stamped_sentances = {}
        letter = 0
        for sentence in sentences:
            tmp = []
            starts = []
            for l in sentence:
                letter += 1
                tmp.append(l)

                i = 0
                for word, start, end in time_stamped:
                    for _ in word:
                        i += 1

                        if i == letter:
                            starts.append(start)
                            starts.append(end)

            letter += 1
            time_stamped_sentances["".join(tmp)] = (min(starts), max(starts))

        return time_stamped_sentances

    def diarize(self, audio_file, min_speakers=1, max_speakers=1):
        diarization = self.diarization_pipeline(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
        speakers_rolls = {}
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"Speaker {speaker}: from {speech_turn.start}s to {speech_turn.end}s")
            speakers_rolls[(speech_turn.start, speech_turn.end)] = speaker

        speakers = set(list(speakers_rolls.values()))
        audio = AudioSegment.from_file(audio_file, format="mp4")
        for speaker in speakers:
            speaker_audio = AudioSegment.empty()
            for key, value in speakers_rolls.items():
                if speaker == value:
                    start = int(round(key[0])) * 1000
                    end = int(round(key[1])) * 1000
                    speaker_audio += audio[start:end]

            speaker_audio.export(self.speakers_audio_dir / f"{speaker}.wav", format="wav")

        self.most_occured_speaker = max(list(speakers_rolls.values()), key=list(speakers_rolls.values()).count)

    def transcribe(self, video_path, use_whisper_fast=False):
        if use_whisper_fast:
            self.model_transcribe = WhisperModel(
                self.whisper_model,
                device="cpu",
            )
            segments, _ = self.model_transcribe.transcribe(self.video_path, word_timestamps=True)
        else:
            self.model_transcribe = whisperx.load_model(self.whisper_model, "cpu", compute_type="float32")
            audio = whisperx.load_audio(video_path)
            result = self.model_transcribe.transcribe(audio, batch_size=1)

            device = "cpu"
            self.model_a, self.metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, device, return_char_alignments=False)
            segments = result["segments"]

        time_stamped_sentances = self.get_time_stamped_setnteces(segments, use_whisper_fast=use_whisper_fast)

        records = []
        for sentence in time_stamped_sentances:
            records.append((sentence,) + time_stamped_sentances[sentence])

        return records

    def to_seconds(self, ts):
        return ts.milliseconds / 1000 + ts.seconds + 60 * ts.minutes + 3600 * ts.hours

    def read_srt(self, srt_path):
        subs = pysrt.open(srt_path)

        records = []
        for sub in subs:
            start = self.to_seconds(sub.start)
            end = self.to_seconds(sub.end)
            records.append((sub.text, start, end))

        return records

    def translate(self, records):
        @torch.inference_mode()
        def trans_fn(sentence):
            inputs = self.mt_tokenizer([sentence], return_tensors="pt", padding=True).to(self.device)
            translated = self.mt_model.generate(**inputs)

            return self.mt_tokenizer.decode(translated[0], skip_special_tokens=True)

        new_records = []
        for text_src, start, end in records:
            text_tgt = trans_fn(sentence=text_src)
            speaker = self.most_occured_speaker
            new_records.append((text_tgt, text_src, start, end, speaker))
            print(new_records[-1])

        return new_records

    def tts(self, records, output_path, start_silence=0.8, theta_min=0.44, theta_max=1):
        audio_chunks_dir = self.output_dir / "audio_chunks"
        audio_chunks_dir.mkdir(parents=True, exist_ok=True)

        su_audio_chunks_dir = self.output_dir / "su_audio_chunks"
        su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)

        _, _, natural_silence, _, _ = records[0]
        previous_silence_time = 0
        if natural_silence >= start_silence:
            previous_silence_time = start_silence
            natural_silence -= start_silence
        else:
            previous_silence_time = natural_silence
            natural_silence = 0

        combined = AudioSegment.silent(duration=natural_silence * 1000)

        audio_files = []
        for i, (text_tgt, _, start, end, speaker) in enumerate(records):
            print('previous_silence_time: ', previous_silence_time)
            fname = f"{i}.wav"
            input_file = audio_chunks_dir / fname
            self.tts_model.tts_to_file(
                text=text_tgt,
                file_path=input_file,
                speaker_wav=self.speakers_audio_dir / f"{speaker}.wav",
                language=self.target_language,
                #speed=2,
            )

            audio = AudioSegment.from_file(input_file)
            current_len = len(audio) / 1000
            original_len =  max(end - start, 0)
            theta = original_len / current_len
            output_file = su_audio_chunks_dir / fname
            if theta_min <= theta < theta_max:
                theta_prim = (original_len + previous_silence_time) / current_len
                command = textwrap.dedent(
                    f"""
                        ffmpeg \
                            -i {str(input_file)} \
                            -filter:a 'atempo={1 / theta_prim}' \
                            -vn \
                            -y \
                            {str(output_file)}
                    """
                )
                process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if process.returncode != 0:
                    sc = original_len + previous_silence_time
                    silence = AudioSegment.silent(duration=(sc * 1000))
                    silence.export(output_file, format="wav")
            elif theta < theta_min:
                silence = AudioSegment.silent(duration=((original_len + previous_silence_time) * 1000))
                silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(previous_silence_time * 1000))
                audio = silence + audio
                audio.export(output_file, format="wav")

            audio = AudioSegment.from_file(output_file)
            current_len = len(audio) / 1000
            original_len = end - start + previous_silence_time
            if i + 1 < len(records):
                _, _, next_start, _, _ = records[i + 1]
                natural_silence = max(next_start - end, 0)
                if natural_silence >= start_silence:
                    previous_silence_time = start_silence
                    natural_silence -= start_silence
                else:
                    previous_silence_time = natural_silence
                    natural_silence = 0

                silence = AudioSegment.silent(duration=((max(original_len - current_len, 0) + natural_silence) * 1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")
            else:
                silence = AudioSegment.silent(duration=(max(original_len - current_len, 0) * 1000))
                audio_with_silence = audio + silence
                audio_with_silence.export(output_file, format="wav")

            print("diff", original_len - current_len)
            print("original_len: ", original_len)
            print("current_len: ", current_len)
            audio_files.append(output_file)

        for audio_file in audio_files:
            audio_segment = AudioSegment.from_file(audio_file)
            combined += audio_segment

        audio = AudioSegment.from_file(self.video_path)
        total_length = len(audio) / 1000
        _, _, _, end, _ = records[-1]
        silence = AudioSegment.silent(duration=abs(total_length - end) * 1000)
        combined += silence
        combined.export(output_path, format="wav")

    def lip_sync(self):
        os.system(
            textwrap.dedent(
                f"""cd Wav2Lip \
                        && \
                        python inference.py \
                            --checkpoint_path '{self.lip_sync_model_name}' \
                            --face '../{str(self.output_dir)}/output_video.mp4' \
                            --audio '../{str(self.audio_dir)}/output.wav' \
                            --face_det_batch_size 1 \
                            --wav2lip_batch_size 1 \
                            --pads 0 15 0 0 \
                            --resize_factor 1 \
                            --nosmooth \
                            --outfile ../{str(self.output_dir)}/result_voice.mp4
                """
            )
        )

    def run(self):
        audio = AudioSegment.from_file(self.video_path, format="mp4")
        audio_file = self.audio_dir / "test0.wav"
        audio.export(audio_file, format="wav")

        if self.huggingface_auth_token:
            self.diarize(audio_file)
        else:
            speaker = "SPEAKER_00"
            audio.export(self.speakers_audio_dir / f"{speaker}.wav", format="wav")
            self.most_occured_speaker = speaker

        if self.use_transcribe:
            if self.srt_path is not None:
                records = self.read_srt(self.srt_path)
            else:
                records = self.transcribe(self.video_path)
        else:
            return

        if self.use_translate:
            records = self.translate(records)
        else:
            return

        if self.use_tts:
            self.tts(records, output_path=self.audio_dir / "output.wav")
        else:
            return

        command = textwrap.dedent(
            f"""
                ffmpeg \
                    -i '{self.video_path}' \
                    -i {str(self.audio_dir)}/output.wav \
                    -c:v copy \
                    -map 0:v:0 \
                    -map 1:a:0 \
                    -shortest \
                    -y \
                    {str(self.output_dir)}/output_video.mp4
            """
        )
        subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if self.use_lip_sync:
            self.lip_sync()


def main():
    args = parse_args()
    set_seed(args.seed)
    nltk.download("punkt")
    load_dotenv()

    VideoDubbing(
        video_path=args.video_url,
        srt_path=args.srt_url,
        output_dir=args.output_dir,
        source_language=args.source_language,
        target_language=args.target_language,
        use_transcribe=args.use_transcribe,
        use_translate=args.use_translate,
        use_tts=args.use_tts,
        use_lip_sync=args.use_lip_sync,
        huggingface_auth_token=os.getenv("HF_TOKEN"),
        whisper_model=args.whisper_model,
        diarization_model_name=args.diarization_model_name,
        tts_model_name=args.tts_model_name,
        lip_sync_model_name=args.lip_sync_model_name,
    ).run()


if __name__ == "__main__":
    main()
