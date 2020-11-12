# pylint: disable=redefined-outer-name, unused-argument
import os
import time
import argparse
import torch
import json
import string

from TTS.utils.synthesis import synthesis
from TTS.utils.generic_utils import load_config, setup_model
from TTS.utils.text.symbols import make_symbols, symbols, phonemes
from TTS.utils.audio import AudioProcessor


def tts(model,
        vocoder_model,
        C,
        VC,
        text,
        ap,
        ap_vocoder,
        use_cuda,
        batched_vocoder,
        speaker_id=None,
        style_wav=None,
        figures=False,
        target=8000,
        overlap=400):
    t_1 = time.time()
    use_vocoder_model = vocoder_model is not None
    waveform, alignment, _, postnet_output, stop_tokens = synthesis(
        model, text, C, use_cuda, ap, speaker_id, style_wav=style_wav,
        truncated=False, enable_eos_bos_chars=C.enable_eos_bos_chars,
        use_griffin_lim=(not use_vocoder_model), do_trim_silence=True)

    if C.model == "Tacotron" and use_vocoder_model:
        postnet_output = ap.out_linear_to_mel(postnet_output.T).T
    # correct if there is a scale difference b/w two models
    if batched_vocoder:
        print('using batched vocoder and target: ', target, ' and overlap: ', overlap)
    if use_vocoder_model:
        #postnet_output = ap._denormalize(postnet_output)
        #postnet_output = ap_vocoder._normalize(postnet_output)
        vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
        waveform = vocoder_model.generate(
            vocoder_input.cuda() if use_cuda else vocoder_input,
            batched=batched_vocoder,
            target=8000,
            overlap=400)
    print(" >  Run-time: {}".format(time.time() - t_1))
    return alignment, postnet_output, stop_tokens, waveform


if __name__ == "__main__":

    global symbols, phonemes

    parser = argparse.ArgumentParser()
    #parser.add_argument('text', type=str, help='Text to generate speech.')
    parser.add_argument('config_path',
                        type=str,
                        help='Path to model config file.')
    #parser.add_argument(
    #    'model_path',
    #    type=str,
    #    help='Path to model file.',
    #)
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save final wav file. Wav file will be names as the text given.',
    )
    parser.add_argument('--use_cuda',
                        type=bool,
                        help='Run model on CUDA.',
                        default=False)
    parser.add_argument(
        '--vocoder_path',
        type=str,
        help=
        'Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
        default="",
    )
    parser.add_argument('--vocoder_config_path',
                        type=str,
                        help='Path to vocoder model config file.',
                        default="")
    parser.add_argument(
        '--batched_vocoder',
        type=bool,
        help="If True, vocoder model uses faster batch processing.",
        default=False)
    parser.add_argument(
        '--target',
        type=int,
        default=8000)
    parser.add_argument(
        '--overlap',
        type=int,
        default=400)
        
    parser.add_argument('--speakers_json',
                        type=str,
                        help="JSON file for multi-speaker model.",
                        default="")
    parser.add_argument(
        '--speaker_id',
        type=int,
        help="target speaker_id if the model is multi-speaker.",
        default=None)
    parser.add_argument(
        '--style_wav_path',
        type=str,
        help="style wav path if the model is GST.",
        default=None)
    args = parser.parse_args()
        
    if args.vocoder_path != "":
        assert args.use_cuda, " [!] Enable cuda for vocoder."
        from WaveRNN.models.wavernn import Model as VocoderModel

    # load the config
    C = load_config(args.config_path)
    C.forward_attn_mask = True
    text = C.sentence
    model_path = C.model_path

    speakers_json = C.speakers_json
    speaker_id = C.speaker_id

    # load the audio processor
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    if 'characters' in C.keys():
        symbols, phonemes = make_symbols(**C.characters)

    # load speakers
    if speakers_json != '':
        speakers = json.load(open(speakers_json, 'r'))
        num_speakers = len(speakers)
    else:
        num_speakers = 0

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C)
    cp = torch.load(model_path)
    model.load_state_dict(cp['model'])
    model.eval()
    if args.use_cuda:
        model.cuda()
    model.decoder.set_r(cp['r'])

    # load vocoder model
    if args.vocoder_path != "":
        VC = load_config(args.vocoder_config_path)
        ap_vocoder = AudioProcessor(**VC.audio)
        
        if VC.bits is not None:
                vocoder_model = VocoderModel(
                        rnn_dims=512,
                        fc_dims=512,
                        bits=VC.bits,
                        pad=VC.pad,
                        upsample_factors=VC.upsample_factors,  # set this depending on dataset
                        feat_dims=80,
                        compute_dims=128,
                        res_out_dims=128,
                        res_blocks=10,
                        hop_length=ap.hop_length,
                        sample_rate=ap.sample_rate
                )
        else:
                vocoder_model = VocoderModel(
                        rnn_dims=512,
                        fc_dims=512,
                        mode=VC.mode,
                        mulaw=VC.mulaw,
                        pad=VC.pad,
                        use_aux_net=VC.use_aux_net,
                        use_upsample_net=VC.use_upsample_net,
                        upsample_factors=VC.upsample_factors,
                        feat_dims=80,
                        compute_dims=128,
                        res_out_dims=128,
                        res_blocks=10,
                        hop_length=ap.hop_length,
                        sample_rate=ap.sample_rate,
                    )

        check = torch.load(args.vocoder_path)
        vocoder_model.load_state_dict(check['model'])
        vocoder_model.eval()
        if args.use_cuda:
            vocoder_model.cuda()
    else:
        vocoder_model = None
        VC = None
        ap_vocoder = None

    # synthesize voice
    print(" > Text: {}".format(text))
    _, _, _, wav = tts(model,
                       vocoder_model,
                       C,
                       VC,
                       text,
                       ap,
                       ap_vocoder,
                       args.use_cuda,
                       args.batched_vocoder,
                       speaker_id=speaker_id,
                       style_wav=args.style_wav_path,
                       figures=False)

    # save the results
    file_name = 'wav_file.wav'
    if vocoder_model is not None:
        file_name = 'wavernn_' + text.replace(" ", "_") + '.wav'
    else:
        file_name = 'gl_' + text.replace(" ", "_") + '.wav'
    out_path = os.path.join(args.out_path, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
