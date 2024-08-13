import torch
import torchaudio
import whisper
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_mel(file_name):
    audio, sample_rate = torchaudio.load(file_name)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
    audio = audio[0]  # get the first channel
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel

def main():
    mel = get_mel("test1.wav").unsqueeze(0)

    print(mel.shape)

    encoder = whisper.load_model("tiny")
    speech_emb = encoder.embed_audio(mel.cuda())

    print(speech_emb)

    ort_session = onnxruntime.InferenceSession("encoder.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(mel)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)


if __name__ == "__main__":
    main()