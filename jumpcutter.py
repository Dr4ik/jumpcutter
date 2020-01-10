import subprocess
from functools import lru_cache

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import re
import math
from shutil import copyfile, rmtree
import os
import argparse
from pytube import YouTube


def download_file(url):
    name = YouTube(url).streams.first().download()
    new_name = name.replace(' ', '_')
    os.rename(name, new_name)
    return new_name


def get_max_volume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv, -minv)


def copy_frame(input_frame, output_frame):
    src = f"{conf.temp_folder}/frame{input_frame + 1:06d}.jpg"
    dst = f"{conf.temp_folder}/newFrame{output_frame + 1:06d}.jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if output_frame % 20 == 19:
        print(f"{output_frame + 1} time-altered frames saved.")
    return True


def input_to_output_filename(file_name):
    dot_index = file_name.rfind(".")
    return f"{file_name[:dot_index]}_ALTERED{file_name[dot_index:]}"


def create_path(s):
    # assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."

    try:
        os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"


def delete_path(s):  # Dangerous! Watch out!
    try:
        rmtree(s, ignore_errors=False)
    except OSError:
        print("Deletion of the directory %s failed" % s)
        print(OSError)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Modifies a video file to play at different speeds when there is sound vs. silence.')
    parser.add_argument('--input_file', type=str, help='the video file you want modified')
    parser.add_argument('--url', type=str, help='A youtube url to download and process')
    parser.add_argument('--output_file', type=str, default="",
                        help="the output file. (optional. if not included, it'll just modify the input file name)")
    parser.add_argument('--silent_threshold', type=float, default=0.03,
                        help="the volume amount that frames' audio needs to surpass to be consider \"sounded\". It ranges from 0 (silence) to 1 (max volume)")
    parser.add_argument('--sounded_speed', type=float, default=1.00,
                        help="the speed that sounded (spoken) frames should be played at. Typically 1.")
    parser.add_argument('--silent_speed', type=float, default=5.00,
                        help="the speed that silent frames should be played at. 999999 for jumpcutting.")
    parser.add_argument('--frame_margin', type=float, default=1,
                        help="some silent frames adjacent to sounded frames are included to provide context. How many frames on either the side of speech should be included? That's this variable.")
    parser.add_argument('--sample_rate', type=float, default=44100, help="sample rate of the input and output videos")
    parser.add_argument('--frame_rate', type=float, default=30,
                        help="frame rate of the input and output videos. optional... I try to find it out myself, but it doesn't always work.")
    parser.add_argument('--frame_quality', type=int, default=3,
                        help="quality of frames to be extracted from input video. 1 is highest, 31 is lowest, 3 is the default.")

    return parser.parse_args()


class Config:
    def __init__(self,
                 frame_rate,
                 sample_rate,
                 silent_threshold,
                 frame_spread,
                 new_speed,
                 url,
                 frame_quality,
                 input_file,
                 output_file,
    ):
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.silent_threshold = silent_threshold
        self.frame_spread = frame_spread
        self.new_speed = new_speed or []
        self.url = url
        self.frame_quality = frame_quality
        self.input_file = input_file
        self.output_file = output_file
        self.temp_folder = "TEMP"
        self.audio_fade_envelope_size = 400  # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)

    @staticmethod
    def from_args(args):
        return Config(
            args.frame_rate,
            args.sample_rate,
            args.silent_threshold,
            args.frame_margin,
            (args.silent_speed, args.sounded_speed),
            args.url,
            args.frame_quality,
            args.input_file if not args.url else download_file(args.url),
            args.output_file if len(args.output_file) >= 1 else input_to_output_filename(args.input_file)
        )


conf = Config.from_args(parse_args())
create_path(conf.temp_folder)


def ffmpeg_scale():
    subprocess.call(
        f"ffmpeg -i {conf.input_file} -qscale:v {conf.frame_quality} {conf.temp_folder}/frame%06d.jpg -hide_banner",
        shell=True
    )


def ffmpeg_extract_audio():
    subprocess.call(
        f"ffmpeg -i {conf.input_file} -ab 160k -ac 2 -ar {conf.sample_rate} -vn {conf.temp_folder}/audio.wav",
        shell=True
    )


def ffmpeg_get_frame_rate():
    with open(f"{conf.temp_folder}/params.txt", "w") as f:
        subprocess.call(f"ffmpeg -i {conf.temp_folder}/input.mp4 2>&1", shell=True, stdout=f)

    with open(f"{conf.temp_folder}/params.txt", 'r+') as f:
        for line in f.readlines():
            m = re.search('Stream #.*Video.* ([0-9]*) fps', line)
            if m is not None:
                return float(m.group(1))

    return None

ffmpeg_scale()
ffmpeg_extract_audio()

class AudioInfo:
    sample_rate = None
    audio_data = None

    def __init__(self,sample_rate, audio_data, frame_rate):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.audio_data = audio_data

    @property
    def audio_sample_count(self): return self.audio_data.shape[0]

    @property
    def max_audio_volume(self): return get_max_volume(self.audio_data)

    @property
    def samples_per_frame(self): return self.sample_rate / self.frame_rate

    @property
    def audio_frame_count(self): return int(math.ceil(self.audio_sample_count / self.samples_per_frame))


@lru_cache()
def get_audio_info(path, frame_rate=None) -> AudioInfo:
    return AudioInfo(*wavfile.read(path), frame_rate or ffmpeg_get_frame_rate())


ai = get_audio_info(f"{conf.temp_folder}/audio.wav")


def get_loud_samples(audio_info: AudioInfo):
    has_loud_audio = np.zeros((audio_info.audio_frame_count,))

    for i in range(audio_info.audio_frame_count):
        start = int(i * audio_info.samples_per_frame)
        end = min(int((i + 1) * audio_info.samples_per_frame), audio_info.audio_sample_count)
        audio_chunks = audio_info.audio_data[start:end]
        max_chunks_volume = float(get_max_volume(audio_chunks)) / audio_info.max_audio_volume
        if max_chunks_volume >= conf.silent_threshold:
            has_loud_audio[i] = 1

    return has_loud_audio


def get_chunks(audio_info: AudioInfo, loud_samples):
    chunks = [[0, 0, 0]]
    should_include_frame = np.zeros((audio_info.audio_frame_count,))
    for i in range(audio_info.audio_frame_count):
        start = int(max(0, i - conf.frame_spread))
        end = int(min(audio_info.audio_frame_count, i + 1 + conf.frame_spread))
        should_include_frame[i] = np.max(loud_samples[start:end])
        if i >= 1 and should_include_frame[i] != should_include_frame[i - 1]:  # Did we flip?
            chunks.append([chunks[-1][1], i, should_include_frame[i - 1]])

    chunks.append([chunks[-1][1], audio_info.audio_frame_count, should_include_frame[audio_info.audio_frame_count - 1]])
    return chunks[1:]

output_audio_data = np.zeros((0, ai.audio_data.shape[1]))
output_pointer = 0

last_existing_frame = None
for chunk in get_chunks(ai, get_loud_samples(ai)):
    audioChunk = ai.audio_data[int(chunk[0] * ai.samples_per_frame):int(chunk[1] * ai.samples_per_frame)]

    sFile = f"{conf.temp_folder}/tempStart.wav"
    eFile = f"{conf.temp_folder}/tempEnd.wav"
    wavfile.write(sFile, conf.sample_rate, audioChunk)
    with WavReader(sFile) as reader:
        with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=conf.new_speed[int(chunk[2])])
            tsm.run(reader, writer)
    _, altered_audio_data = wavfile.read(eFile)
    leng = altered_audio_data.shape[0]
    end_pointer = output_pointer + leng
    output_audio_data = np.concatenate((output_audio_data, altered_audio_data / ai.max_audio_volume))

    if leng < conf.audio_fade_envelope_size:
        output_audio_data[output_pointer:end_pointer] = 0  # audio is less than 0.01 sec, let's just remove it.
    else:
        pre_mask = np.arange(conf.audio_fade_envelope_size) / conf.audio_fade_envelope_size
        mask = np.repeat(pre_mask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
        output_audio_data[output_pointer:output_pointer + conf.audio_fade_envelope_size] *= mask
        output_audio_data[end_pointer - conf.audio_fade_envelope_size:end_pointer] *= 1 - mask

    start_output_frame = int(math.ceil(output_pointer / ai.samples_per_frame))
    endOutputFrame = int(math.ceil(end_pointer / ai.samples_per_frame))
    for output_frame in range(start_output_frame, endOutputFrame):
        input_frame = int(chunk[0] + conf.new_speed[int(chunk[2])] * (output_frame - start_output_frame))
        didItWork = copy_frame(input_frame, output_frame)
        if didItWork:
            last_existing_frame = input_frame
        else:
            copy_frame(last_existing_frame, output_frame)

    output_pointer = end_pointer

wavfile.write(f"{conf.temp_folder}/audioNew.wav", conf.sample_rate, output_audio_data)

'''
output_frame = math.ceil(output_pointer/samples_per_frame)
for endGap in range(output_frame,audio_frame_count):
    copyFrame(int(audio_sample_count/samples_per_frame)-1,endGap)
'''

command = f"ffmpeg -framerate {self.frame_rate} -i {conf.temp_folder}/newFrame%06d.jpg -i {conf.temp_folder}/audioNew.wav -strict -2 {self.output_file}"
subprocess.call(command, shell=True)

delete_path(conf.temp_folder)
