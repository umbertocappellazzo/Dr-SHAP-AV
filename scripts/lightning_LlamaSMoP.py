#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:04:39 2024

@author: umbertocappellazzo
"""
import sys
sys.path.append("..")
from dataclasses import dataclass
import numpy as np
import torch
import torchaudio
from utils.cosine import WarmupCosineScheduler
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from .Llama_LoRA import LoRA_config
from .modeling_LlamaSMoP import AVSR_LLMs
from tokenizers.processors import TemplateProcessing
import os

DEFAULT_PAD_TOKEN = "<pad>"
AUDIO_SOS = "<audio>"
AUDIO_EOS = "</audio>"
VIDEO_SOS = "<video>"
VIDEO_EOS = "</video>"


llm_size = {"meta-llama/Llama-3.2-1B": 2048,
            "meta-llama/Llama-3.2-3B": 3072
            }


@dataclass
class MoP_config_class:
    MoP_experts: int
    top_k: int
    apply_load_balancing_loss: bool = False
    apply_router_z_loss: bool = False

def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)

class ModelModule_LLM(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        if args.use_lora_avhubert:
            assert "lora_avhubert" in args.unfrozen_modules, ("LoRA modules for the AV-HuBERT encoder must be unfrozen!!")

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, add_bos_token=True, add_eos_token= True)
        
        # Apparently, some LLMs don't rely on FastTokenizer and it seems like they don't append the EOS token even though you set
        # it explicitly. In my case, this happens for LLama3. More details at: https://github.com/huggingface/transformers/issues/22794.

        bos = self.tokenizer.bos_token
        eos = self.tokenizer.eos_token

        self.tokenizer._tokenizer.post_processor =TemplateProcessing(
            single=f"{bos}:0 $A:0 {eos}:0",
            pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
            special_tokens=[
                (f"{bos}", self.tokenizer.bos_token_id),
                (f"{eos}", self.tokenizer.eos_token_id)
            ],
        )
        
        # By default, LLaMA doesn't come with a padding token (pad_token= None), so we need to introduce it.
        num_added_toks = self.tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN, "additional_special_tokens": [AUDIO_SOS, AUDIO_EOS, VIDEO_SOS, VIDEO_EOS]})
        pad_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
            
        print("We have added ", num_added_toks, " tokens to the tokenizer!")
        self.tokenizer.padding_side = "right"   # The padding is added to the right.

        if args.modality == 'audio':
            prompt = args.prompt_audio
        elif args.modality == 'video':
            prompt = args.prompt_video
        else:
            assert args.modality in ['audiovisual', 'audiovisual_avhubert']
            prompt = args.prompt_audiovisual
        
        print(f"The prompt used for the {args.modality} modality is: {prompt}")
        
        if args.add_PEFT_LLM:
            IS_LLAMA3_2_3B = True if args.llm_model == "meta-llama/Llama-3.2-3B" else False
                
            lora_config_llm = LoRA_config(args.rank, args.alpha, IS_LLAMA3_2_3B)
            MoP_config = MoP_config_class(MoP_experts = args.MoP_experts, top_k = args.MoP_topk, apply_load_balancing_loss = args.apply_load_balancing_loss, apply_router_z_loss = args.apply_router_z_loss)
                
            self.model = AVSR_LLMs(modality = args.modality,
                                   pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path, 
                                   use_lora_avhubert= args.use_lora_avhubert,
                                   llm_model = args.llm_model, 
                                   hidden_size = llm_size[args.llm_model], 
                                   intermediate_size= args.intermediate_size, 
                                   tokenizer = self.tokenizer, 
                                   prompt = prompt, 
                                   pad_id = pad_id, 
                                   downsample_ratio_audio = args.downsample_ratio_audio, 
                                   downsample_ratio_video = args.downsample_ratio_video, 
                                   audio_encoder_name = args.audio_encoder_name,
                                   unfrozen_modules= args.unfrozen_modules, 
                                   max_dec_tokens = args.max_dec_tokens, 
                                   num_beams = args.num_beams, 
                                   PEFT_LLM_name = args.add_PEFT_LLM,
                                   peft_config_llm= lora_config_llm,
                                   MoP_config = MoP_config,
                                   )

            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)
                
        
        # initialize the full model from the checkpoint for inference.
        if args.pretrained_model_path:
            ckpt = torch.load(args.pretrained_model_path)
            self.model.load_state_dict(ckpt) # strict = False
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params= self.model.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        llm_loss, load_balancing_loss, router_z_loss = self.model(batch, is_trainval = True)
        train_loss = llm_loss + self.args.load_balancing_loss_coeff*load_balancing_loss + self.args.router_z_loss_coeff*router_z_loss
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss", train_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("llm_loss", llm_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("load_balancing_loss", load_balancing_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("router_z_loss", router_z_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        
        batch_sizes = self.all_gather(batch_size)
        
        train_loss *= batch_sizes.size(0) / batch_sizes.sum()
        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
        
        return train_loss
            
    def validation_step(self, batch, batch_idx):
        
        llm_loss_val, load_balancing_loss_val, router_z_loss_val = self.model(batch, is_trainval = True)
        loss_val = llm_loss_val + self.args.load_balancing_loss_coeff*load_balancing_loss_val + self.args.router_z_loss_coeff*router_z_loss_val
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss_val", loss_val, batch_size=batch_size, sync_dist=True)
        self.log("load_balancing_loss_val", load_balancing_loss_val, batch_size=batch_size, sync_dist=True)
        self.log("router_z_loss_val", router_z_loss_val, batch_size=batch_size, sync_dist=True)
        
        return loss_val
    
    def test_step(self, batch, batch_idx):
        
        batch_size = batch["tokens"].shape[0]
        if self.args.compute_shap:
            audio_shap_abs_current, video_shap_abs_current, num_audio_tokens, shapley_values = self.model.forward_shap(batch, nsamples=self.args.num_samples_shap, shap_alg = self.args.shap_alg)

            self.audio_shap_abs.append(audio_shap_abs_current)
            self.video_shap_abs.append(video_shap_abs_current)
            self.num_audio_tokens.append(num_audio_tokens)
            self.shapley_values.append(shapley_values)

            self.log("sample-audio-ABS-SHAP", audio_shap_abs_current, on_step=True, on_epoch=False, batch_size=batch_size, prog_bar=False)
            self.log("sample-video-ABS-SHAP", video_shap_abs_current, on_step=True, on_epoch=False, batch_size=batch_size, prog_bar=False)
            self.log("sample-num-audio-tokens", num_audio_tokens, on_step=True, on_epoch=False, batch_size=batch_size, prog_bar=False)

        
        generated_ids = self.model(batch, is_trainval = False)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # If we want to inspect what the model predicts and compare with the ground truth, uncomment the two lines below.
        #print("Input text: ", batch["gold_text"])
        #print("Generated text: ", generated_text)

        self.total_edit_distance += compute_word_level_distance(batch["gold_text"], generated_text)
        self.total_length += len(batch["gold_text"].split())
        
        return

    
    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0

        if self.args.compute_shap:
            self.audio_shap_abs = []
            self.video_shap_abs = []
            self.num_audio_tokens = []
            self.shapley_values = []

            if self.args.output_path_shap is not None:
                self.output_file = os.path.join(
                    self.args.output_path_shap,
                    self.args.exp_name

                )
            print("Output dir: ", self.output_file)
        
    def on_test_epoch_end(self):
        
        self.log("wer", self.total_edit_distance / self.total_length)
        if self.args.compute_shap:
            overall_audio_abs = np.mean(self.audio_shap_abs)
            overall_video_abs = np.mean(self.video_shap_abs)
            overall_num_audio_tokens = np.mean(self.num_audio_tokens)

            std_overall_audio_abs = np.std(self.audio_shap_abs)
            std_overall_video_abs = np.std(self.video_shap_abs)

            self.log("audio-ABS-SHAP", overall_audio_abs)
            self.log("video-ABS-SHAP", overall_video_abs)
            self.log("STD_audio-ABS-SHAP", std_overall_audio_abs)
            self.log("STD_video-ABS-SHAP", std_overall_video_abs)
            self.log("num-audio-tokens", overall_num_audio_tokens)

            print("Global Audio-ABS-SHAP :", overall_audio_abs * 100, "%")
            print("Global Video-ABS-SHAP :", overall_video_abs * 100, "%")

            if self.args.output_path_shap is not None:
                np.savez_compressed(
                        self.output_file,
                        audio_abs=np.array(self.audio_shap_abs),
                        video_abs=np.array(self.video_shap_abs),
                        audio_pos=np.array(self.audio_shap_pos),
                        video_pos=np.array(self.video_shap_pos),
                        audio_neg=np.array(self.audio_shap_neg),
                        video_neg=np.array(self.video_shap_neg),
                        num_audio_tokens=np.array(self.num_audio_tokens),
                        shap_values=np.array(self.shapley_values, dtype=object),
                    )