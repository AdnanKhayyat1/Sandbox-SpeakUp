from resemblyzer import preprocess_wav, VoiceEncoder
from spectralcluster import SpectralClusterer
from pathlib import Path
import librosa
import numpy as np
from scipy.io.wavfile import write
from scipy.io import wavfile
import os
import speech_recognition as sr
import time
import azure.cognitiveservices.speech as speechsdk

tts_key = "5e658863795544bc976f49716c1659d2"
region = "westus"

encoder = VoiceEncoder("cpu")
r = sr.Recognizer()
clusterer = SpectralClusterer(
    min_clusters=1,
    max_clusters=10,
    # p_percentile=0.90,
    # gaussian_blur_sigma=1
)

def create_labelling(labels,wav_splits):
    sampling_rate = 16000
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0
    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))
    return labelling
audio = "test.wav"

def asr(audio):
    speech_configuration = speechsdk.SpeechConfig(subscription=tts_key, region=region)
    audio_configuration = speechsdk.audio.AudioConfig(filename=audio)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_configuration, audio_config=audio_configuration)
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized speech: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("Speech not recognized: {}".format(result.no_match_details))
    return result.text

if __name__ == '__main__':
    audio_file_path = 'file.wav'
    wav_fpath = Path(audio_file_path)

    # iterate an arbitrary number of maximum steps
    for j in range(1200):
        start = time.time()

        # tries to read and preprocess wav file (sometimes fails if file is being written to)
        try:
            wav = preprocess_wav(wav_fpath)
        except:
            continue

        # runs diarization
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
        labels = clusterer.predict(cont_embeds)
        labelling = create_labelling(labels, wav_splits)

        # writes new wave that has been trimmed for silence
        write('file_trimmed.wav', 16000, wav)

        # empties split_wavs directory
        dir = 'split_wavs/'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

        # reads in and resamples file that has had silence trimmed out
        rate, data = wavfile.read('file_trimmed.wav')
        split_wavs = []
        num_frames_removed = 0

        # slices and saves new wav files to split_wavs directory
        for i in range(len(labelling)):
            if labelling[i][2] - labelling[i][1] > .6:
                split_at_frame = int(rate * labelling[i][2]) - num_frames_removed
                left_data, data = data[:split_at_frame - 1], data[split_at_frame:]
                num_frames_removed += len(left_data)
                wavfile.write('split_wavs/file_' + str(i) + '.wav', rate, left_data) # (left_data * maxv).astype(np.int16)

        # performs ASR and appends speaker's words to diarization
        diarization = []
        files = os.listdir(dir)[::-1]
        for f in files:
            text = asr(dir+f)
            diarization.append('New Speaker: ' + text)
        print('\n'.join(diarization))
        print(time.time() - start)


