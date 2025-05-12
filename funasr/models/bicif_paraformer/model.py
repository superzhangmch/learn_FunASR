#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import copy
import time
import torch
import logging
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict, List, Optional, Tuple

from funasr.register import tables
from funasr.models.ctc.ctc import CTC
from funasr.utils import postprocess_utils
from funasr.metrics.compute_acc import th_accuracy
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.paraformer.model import Paraformer
from funasr.models.paraformer.search import Hypothesis
from funasr.train_utils.device_funcs import force_gatherable
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.train_utils.device_funcs import to_device

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "BiCifParaformer")
class BiCifParaformer(Paraformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paper1: FunASR: A Fundamental End-to-End Speech Recognition Toolkit
    https://arxiv.org/abs/2305.11013
    Paper2: Achieving timestamp prediction while recognizing with non-autoregressive end-to-end ASR model
    https://arxiv.org/abs/2301.12343
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _calc_pre2_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        _, _, _, _, pre_token_length2 = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id
        )

        # loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)
        loss_pre2 = self.criterion_pre(ys_pad_lens.type_as(pre_token_length2), pre_token_length2)

        return loss_pre2

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, ys_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_pad_lens = ys_pad_lens + self.predictor_bias
        pre_acoustic_embeds, pre_token_length, _, pre_peak_index, _ = self.predictor(
            encoder_out, ys_pad, encoder_out_mask, ignore_id=self.ignore_id
        )

        # 0. sampler
        decoder_out_1st = None
        if self.sampling_ratio > 0.0:
            sematic_embeds, decoder_out_1st = self.sampler(
                encoder_out, encoder_out_lens, ys_pad, ys_pad_lens, pre_acoustic_embeds
            )
        else:
            sematic_embeds = pre_acoustic_embeds

        # 1. Forward decoder
        decoder_outs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_out, _ = decoder_outs[0], decoder_outs[1]

        if decoder_out_1st is None:
            decoder_out_1st = decoder_out
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_pad)
        acc_att = th_accuracy(
            decoder_out_1st.view(-1, self.vocab_size),
            ys_pad,
            ignore_label=self.ignore_id,
        )
        loss_pre = self.criterion_pre(ys_pad_lens.type_as(pre_token_length), pre_token_length)

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out_1st.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, loss_pre

    def calc_predictor(self, encoder_out, encoder_out_lens):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index, pre_token_length2 = (
            self.predictor(encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)
        )
        return pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index

    def calc_predictor_timestamp(self, encoder_out, encoder_out_lens, token_num):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        ds_alphas, ds_cif_peak, us_alphas, us_peaks = self.predictor.get_upsample_timestamp(
            encoder_out, encoder_out_mask, token_num
        )
        return ds_alphas, ds_cif_peak, us_alphas, us_peaks

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_ctc, cer_ctc = None, None
        loss_pre = None
        stats = dict()

        # decoder: CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # decoder: Attention decoder branch
        loss_att, acc_att, cer_att, wer_att, loss_pre = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )

        loss_pre2 = self._calc_pre2_loss(encoder_out, encoder_out_lens, text, text_lengths)

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = (
                loss_att
                + loss_pre * self.predictor_weight
                + loss_pre2 * self.predictor_weight * 0.5
            )
        else:
            loss = (
                self.ctc_weight * loss_ctc
                + (1 - self.ctc_weight) * loss_att
                + loss_pre * self.predictor_weight
                + loss_pre2 * self.predictor_weight * 0.5
            )

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
        stats["loss_pre2"] = loss_pre2.detach().cpu()

        stats["loss"] = torch.clone(loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + self.predictor_bias).sum())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        # init beamsearch
        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc != None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )
        if self.beam_search is None and (is_use_lm or is_use_ctc):
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

        meta_data = {}
        # if isinstance(data_in, torch.Tensor):  # fbank
        #     speech, speech_lengths = data_in, data_lengths
        #     if len(speech.shape) < 3:
        #         speech = speech[None, :, :]
        #     if speech_lengths is None:
        #         speech_lengths = speech.shape[1]
        # else:
        # extract fbank feats
        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(
            data_in, fs=frontend.fs, audio_fs=kwargs.get("fs", 16000)
        )
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        speech, speech_lengths = extract_fbank(
            audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
        )
        time3 = time.perf_counter()
        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        meta_data["batch_data_time"] = (
            speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
        )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        # print (encoder_out.shape, encoder_out_lens)
        # torch.Size([3, 84, 512]) tensor([45, 77, 84], device='cuda:0') # note: 84 = max(45, 77, 84)
        # 上面输出是在加载 modelscope 下的 model：modelscope/hub/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
        # 对某个 audio【该modelscope model 下的 example/asr_example.wav】作为input的结果。下面都基于此. 原始 input audio 被切分成了3个不等长的片段，内容分别是：
        # - 因为如果当你认为这个世界没有正义， 
        # - 正是因为存在绝对正义所以我们接受现实的相对正义
        # - 但是不要因为>现实的相对正义我们就认为这个世界没有正义
        # 也就是说，是按自然语音停顿分隔 (VAD) 做的切分
    
        # predictor
        predictor_outs = self.calc_predictor(encoder_out, encoder_out_lens)
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = (
            predictor_outs[0],
            predictor_outs[1],
            predictor_outs[2],
            predictor_outs[3],
        )

        # print (pre_acoustic_embeds.shape, pre_token_length.shape, alphas.shape, pre_peak_index.shape)
        # 输出： torch.Size([3, 27, 512]) torch.Size([3]) torch.Size([3, 85]) torch.Size([3, 85])
        # [3, 27, 512] == (batch_size, seq_len, dim_feature)。 seq_len=27 表示 27 个token。
        # [3, 85] 中的 85，对应前面的 max([45, 77, 84]) + 1

        pre_token_length = pre_token_length.round().long()
        if torch.max(pre_token_length) < 1:
            return []
        decoder_outs = self.cal_decoder_with_predictor(
            encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length
        )
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]
        # print (decoder_out.shape, ys_pad_lens)
        # torch.Size([3, 27, 8404]) tensor([17, 24, 27], device='cuda:0')
        # 三个子句分别有 17， 24， 27 个 tokens。8404 是词表大小（model_name=speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch中也有8404）
        # asr 一个难点就是语音和文字的对齐问题，所以才发明了 CTC。而 funASR 这样的方法巧妙之处就在于预测出了每个token的语音长度，先预测出了有多少个token，从而解决了对齐问题。

        # BiCifParaformer, test no bias cif2
        _, _, us_alphas, us_peaks = self.calc_predictor_timestamp(
            encoder_out, encoder_out_lens, pre_token_length
        )

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            # 遍历每一个子句
            x = encoder_out[i, : encoder_out_lens[i], :]
            am_scores = decoder_out[i, : pre_token_length[i], :]

            # beam search or greedy search
            if self.beam_search is not None:
                # beam search：注意，如果用到它，则要么是用到了ctc，要么是用到了 ngram 之类语言模型的辅助。否则 funASR 本身的方法是没法用beam search的（因为 greedy search 已经是最优解了）
                nbest_hyps = self.beam_search(
                    x=x,
                    am_scores=am_scores,
                    maxlenratio=kwargs.get("maxlenratio", 0.0),
                    minlenratio=kwargs.get("minlenratio", 0.0),
                )

                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                # greedy 搜索。默认分支
                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)] # Hypothesis并没干啥特别的

            for nbest_idx, hyp in enumerate(nbest_hyps):
                # 注意 hyp 已经有结果 token id 了，只差反解出 text 了
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # remove blank symbol id, which is assumed to be 0
                token_int = list(
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                if tokenizer is not None:
                    # Change integer-ids to tokens
                    token = tokenizer.ids2tokens(token_int) # 每个token的text
                    text = tokenizer.tokens2text(token) # 把 token_arr 拼接成最终 text

                    _, timestamp = ts_prediction_lfr6_standard(  # 这个函数内部，其实就是：利用 CIF 模型中每个 token 的 fire point（触发点），来界定该 token 在原始音频中的时间偏移（time offset）和持续时间。
                        us_alphas[i][: encoder_out_lens[i] * 3], # 其实正是从 alphas 推断出的 us_peaks
                        us_peaks[i][: encoder_out_lens[i] * 3],  # fire 点
                        copy.copy(token),
                        vad_offset=kwargs.get("begin_time", 0),
                    )

                    text_postprocessed, time_stamp_postprocessed, word_lists = (
                        postprocess_utils.sentence_postprocess(token, timestamp)
                    )

                    result_i = {
                        "key": key[i],
                        "text": text_postprocessed,
                        "timestamp": time_stamp_postprocessed,
                    }

                    if ibest_writer is not None:
                        ibest_writer["token"][key[i]] = " ".join(token)
                        # ibest_writer["text"][key[i]] = text
                        ibest_writer["timestamp"][key[i]] = time_stamp_postprocessed
                        ibest_writer["text"][key[i]] = text_postprocessed
                else:
                    result_i = {"key": key[i], "token_int": token_int}
                results.append(result_i)

        return results, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models
