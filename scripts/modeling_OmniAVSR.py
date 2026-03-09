#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:58:09 2026

@author: umbertocappellazzo
"""

import sys
sys.path.append("..")
import torch
from torch import nn
from .Llama_LoRA import LlamaForCausalLM_lora
from transformers import WhisperModel, LlamaForCausalLM, AutoFeatureExtractor
import fairseq
from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq, AVHubertSeq2SeqConfig
from av_hubert.avhubert.hubert_lora import AVHubertModel_lora
import math
import random

import numpy as np
import shap

#from av_hubert.avhubert.hubert import AVHubertModel
#from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq
#from AV_HuBERT_encoder import avhubertConfig, avhubert_encoder

IGNORE_INDEX = -100

class AVSR_LLMs(nn.Module):
    def __init__(self, modality, pretrain_avhubert_enc_video, use_lora_avhubert, llm_model, hidden_size, 
                 intermediate_size, tokenizer, prompt_audio, prompt_video, prompt_audiovisual, pad_id, 
                 downsample_ratio_audio, downsample_ratio_video, audio_encoder_name,
                 unfrozen_modules, max_dec_tokens, num_beams, PEFT_LLM_name = None, peft_config_llm = None,
                 matry_weights = None, is_task_specific = None
                 ):
        
        super().__init__()
        
        self.modality = modality
        self.pretrain_avhubert_enc_video = pretrain_avhubert_enc_video
        self.max_dec_tokens = max_dec_tokens
        self.num_beams = num_beams
        self.downsample_ratio_audio = downsample_ratio_audio
        self.downsample_ratio_video = downsample_ratio_video
        self.audio_encoder_name = audio_encoder_name
        self.llm_model = llm_model
        self.peft_config_llm = peft_config_llm
        self.PEFT_LLM_name = PEFT_LLM_name
        self.hidden_size = hidden_size
        self.matry_weights = matry_weights
        self.is_task_specific = is_task_specific
            
        if modality == "audio" or modality == "audiovisual":
                
            print("Instantiating whisper!")    
            self.audio_encoder = WhisperModel.from_pretrained(self.audio_encoder_name).encoder
            self.audio_frontend = AutoFeatureExtractor.from_pretrained(self.audio_encoder_name)
            self.audio_encoder.requires_grad_(False)
            self.audio_encoder.train() # This must be explicitly done as by default the from_pretrained HF models are in eval mode when initialized (this is the opposite for pytorch!)--> cause a break in deepspeed 3! https://github.com/Lightning-AI/pytorch-lightning/issues/19467
            audio_dim =self.audio_encoder.config.hidden_size
            
                
            self.matry_map_audio = {}
            for index, el in enumerate(self.downsample_ratio_audio):
                self.matry_map_audio[el] = index
            
            print("Instantiating avg-pooling projector for audio Matryoshka!")
            self.avg_pool_audio = nn.ModuleList([nn.AvgPool1d(downsample_ratio) for downsample_ratio in self.downsample_ratio_audio])
            self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size, nn.LayerNorm(hidden_size))) for _ in self.downsample_ratio_audio])
           
        if modality == "video" or modality == "audiovisual":
            assert self.pretrain_avhubert_enc_video is not None, "You must specify pretrain_avhubert_enc_video argument!"
            print("Initializing AV-HuBERT Large, non fine-tuned!")
            
            if use_lora_avhubert:
                
                modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_video])
                self.video_encoder = modell[0]
                
                print('Preparing LoRA layers for AV-HuBERT video-only!')
                
                
                num_layers_avhubert = 24 if "large" in self.pretrain_avhubert_enc_video else 12
                video_dim_avhubert = 1024 if "large" in self.pretrain_avhubert_enc_video else 768
                for layer_idx in range(num_layers_avhubert):
                    # We set apply_lora = True for each video encoder layer such that it is applied. TODO: define this parameter in the AV-HuBERT main class.
                    self.video_encoder.encoder.layers[layer_idx].apply_lora = True
                    
                    self.video_encoder.encoder.layers[layer_idx].self_attn.rank = 16
                    self.video_encoder.encoder.layers[layer_idx].self_attn.scaling_lora = 2
                    
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q = nn.Linear(video_dim_avhubert, round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), bias= False)
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q = nn.Linear(round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), video_dim_avhubert, bias= False)
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_V = nn.Linear(video_dim_avhubert, round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), bias= False)
                    self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_V = nn.Linear(round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), video_dim_avhubert, bias= False)
        
                    nn.init.zeros_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q.weight)
                    nn.init.zeros_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_V.weight)
                    nn.init.kaiming_uniform_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q.weight, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_V.weight, a=math.sqrt(5))
                
            else:
                modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_video])
                self.video_encoder = modell[0]
                
            self.video_encoder.requires_grad_(False)
            video_dim = 1024 if "large" in self.pretrain_avhubert_enc_video else 768
                
                
            self.matry_map_video = {}
            for index, el in enumerate(self.downsample_ratio_video):
                self.matry_map_video[el] = index
            
            print("Instantiating avg-pooling projector for video Matryoshka!")
            self.avg_pool_video = nn.ModuleList([nn.AvgPool1d(downsample_ratio) for downsample_ratio in self.downsample_ratio_video])
            self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size, nn.LayerNorm(hidden_size))) for _ in self.downsample_ratio_video])
            
        if self.PEFT_LLM_name is None:
            self.llm = LlamaForCausalLM.from_pretrained(llm_model)
        else:
            assert self.PEFT_LLM_name == "lora"
            self.llm = LlamaForCausalLM_lora.from_pretrained(llm_model, peft_config_llm) 
               
        
        # IMPORTANT: we need to add the pad_id to the model and resize the token embeddings matrix accordingly.
        self.tokenizer = tokenizer
        self.llm.config.pad_token_id = pad_id
        
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.llm.requires_grad_(False)
        
        prompt_tokens_start_at = 1
        self.register_buffer("prompt_audio", self.llm.model.embed_tokens(self.tokenizer(prompt_audio, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1]))
        self.register_buffer("prompt_video", self.llm.model.embed_tokens(self.tokenizer(prompt_video, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1]))
        self.register_buffer("prompt_audiovisual", self.llm.model.embed_tokens(self.tokenizer(prompt_audiovisual, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1]))
        
        self.prompt_audio_len = self.prompt_audio.shape[1]
        self.prompt_video_len = self.prompt_video.shape[1]
        self.prompt_audiovisual_len = self.prompt_audiovisual.shape[1]
        
        print(f"The audio prompt has {self.prompt_audio_len} tokens.")
        print(f"The video prompt has {self.prompt_video_len} tokens.")
        print(f"The audiovisual prompt has {self.prompt_audiovisual_len} tokens.")
        
        self._unfreeze_PEFT(unfrozen_modules)
        
        
    def _unfreeze_PEFT(self, unfrozen_modules):
        """
        Modules to be unfrozen. Unfrozen_blocks is a list with one or multiple values: ['peft_audio','peft_video','embedding','peft_llm']. 
        """
        if None in unfrozen_modules:
            return
        if "peft_llm" in unfrozen_modules:
            print("Unfreezing LoRA for LLM:")
            for block_idx in range(self.llm.config.num_hidden_layers):
                self.llm.model.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
                
                if self.peft_config_llm.SHARED_LORA:
                    self.llm.model.layers[block_idx].self_attn.lora_down_Q_shared.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_up_Q_shared.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_down_V_shared.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_up_V_shared.requires_grad_(True)
        
        if "lora_avhubert" in unfrozen_modules:
            print("Unfreezing LoRA for AV-HuBERT video encoder!")
            for block_idx in range(24):
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
            
        
    def f_shap(
        self,
        mask,
        audio_tokens,
        video_tokens,
        full_baseline_emb,
        llm,
        baseline_ids
    ):
        """
        Computes f(S) for a coalition S by removing features from the full baseline.
        
        """
        device = full_baseline_emb.device
        mask = torch.tensor(mask, dtype=torch.bool, device=device)
        
        N_a = audio_tokens.shape[1] if audio_tokens is not None else 0
        N_v = video_tokens.shape[1] if video_tokens is not None else 0
        
        # Clone the FULL baseline (all features present)
        masked_emb = full_baseline_emb.clone()
        
        # Extract masks - AUDIO first, VIDEO second (matching embedding order)
        mask_audio = mask[:N_a] if N_a > 0 else torch.tensor([], dtype=torch.bool, device=device)
        mask_video = mask[N_a:] if N_v > 0 else torch.tensor([], dtype=torch.bool, device=device)
        
        # Remove audio features where mask = 0
        if N_a > 0 and (~mask_audio).any():
            a_start, a_end = self.audio_content_start, self.audio_content_end
            audio_section = masked_emb[:, a_start:a_end, :].clone()
            audio_section[:, ~mask_audio, :] = 0.0
            masked_emb[:, a_start:a_end, :] = audio_section
        
        # Remove video features where mask = 0
        if N_v > 0 and (~mask_video).any():
            v_start, v_end = self.video_content_start, self.video_content_end
            video_section = masked_emb[:, v_start:v_end, :].clone()
            video_section[:, ~mask_video, :] = 0.0
            masked_emb[:, v_start:v_end, :] = video_section
        
        # Append baseline answer tokens as embeddings
        baseline_ids = baseline_ids.to(device)
        baseline_emb = self.llm.model.embed_tokens(baseline_ids.unsqueeze(0))
        concat_emb = torch.cat([masked_emb, baseline_emb], dim=1)
        
        # Forward pass to get logits
        outputs = llm(
            inputs_embeds=concat_emb,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            modality = self.modality
        )
        
        logits = outputs.logits.squeeze(0)
        L_prefix = masked_emb.shape[1]
        T = baseline_ids.shape[0]
        
        # Extract logits for baseline tokens
        positions = L_prefix + torch.arange(T, device=device)
        logit_vec = logits[positions, baseline_ids]
        
        return logit_vec.detach().cpu().numpy()
    
    
    def shap_wrapper(
        self, 
        masks,
        audio_tokens,
        video_tokens,
        full_baseline_emb,
        llm,
        baseline_ids
    ):
        """
        Wrapper to handle both single masks and batches of masks for SHAP.
        """
        if masks.ndim == 1:
            return self.f_shap(
                masks,
                audio_tokens,
                video_tokens,
                full_baseline_emb,
                llm,
                baseline_ids
            )
        else:
            return np.array([
                self.f_shap(
                    m,
                    audio_tokens,
                    video_tokens,
                    full_baseline_emb,
                    llm,
                    baseline_ids
                )
                for m in masks
            ])
        
        
        
    def forward_shap(self, inputs, ratio_audio, ratio_video, nsamples, shap_alg):
        """
        Compute SHAP values for audio and video contributions to LLM-based AVSR.
        Uses "remove from full" approach: mask=1 means keep, mask=0 means remove.
        
        """
        
        # Encode audio/video
        audio_features, num_audio_tokens = self.encode_audio(inputs["audio"], max(inputs["lengths"]), test_ratio_matry_audio = ratio_audio, return_num_tokens=True)
        video_features = self.encode_video(inputs["video"], test_ratio_matry_video = ratio_video)
    
        device = inputs["tokens"].device
    
        N_a = audio_features.shape[1]
        N_v = video_features.shape[1]
        p = N_a + N_v
    
        # Build prompt embeddings
        text_embeddings_raw = self.llm.model.embed_tokens(inputs["tokens"])
        
        prompt_embeddings = self.prompt_audiovisual
        
        
        BOS = text_embeddings_raw[:, 0, :].unsqueeze(1)
    
        # Prepare special token embeddings
        vid_S = self.llm.model.embed_tokens(
            torch.tensor([self.tokenizer.vocab["<video>"]], device=device).expand(1, -1)
        )
        vid_E = self.llm.model.embed_tokens(
            torch.tensor([self.tokenizer.vocab["</video>"]], device=device).expand(1, -1)
        )
    
        aud_S = self.llm.model.embed_tokens(
            torch.tensor([self.tokenizer.vocab["<audio>"]], device=device).expand(1, -1)
        )
        aud_E = self.llm.model.embed_tokens(
            torch.tensor([self.tokenizer.vocab["</audio>"]], device=device).expand(1, -1)
        )
    
        # Project features
        
        video_tokens = self.video_proj[self.matry_map_video[ratio_video]](video_features)
        audio_tokens = self.audio_proj[self.matry_map_audio[ratio_audio]](audio_features)
        
        #audio_tokens = torch.zeros_like(audio_tokens)
        
        #video_tokens = torch.zeros_like(video_tokens)
    
        # Build FULL baseline with ALL audio and video features
        # Order: BOS → AUDIO → VIDEO → PROMPT (matching training)
        blocks = []
        idx = 0
        
        blocks.append(BOS)
        idx += 1
    
        # AUDIO BLOCK (first, as in training)
        if audio_tokens is not None:
            blocks.append(aud_S)
            idx += 1
            audio_content_start = idx
            blocks.append(audio_tokens)
            idx += N_a
            audio_content_end = idx
            blocks.append(aud_E)
            idx += 1
        else:
            audio_content_start = audio_content_end = None
    
        # VIDEO BLOCK (second, as in training)
        if video_tokens is not None:
            blocks.append(vid_S)
            idx += 1
            video_content_start = idx
            blocks.append(video_tokens)
            idx += N_v
            video_content_end = idx
            blocks.append(vid_E)
            idx += 1
        else:
            video_content_start = video_content_end = None
    
        blocks.append(prompt_embeddings)
    
        full_baseline_emb = torch.cat(blocks, dim=1)
    
        # Store content positions for f_shap()
        self.video_content_start = video_content_start
        self.video_content_end = video_content_end
        self.audio_content_start = audio_content_start
        self.audio_content_end = audio_content_end
    
        # Generate baseline with ALL features
        baseline_ids = self.llm.generate(
            inputs_embeds=full_baseline_emb,
            max_new_tokens=self.max_dec_tokens,
            num_beams=self.num_beams,
            eos_token_id=self.tokenizer.vocab["<|end_of_text|>"],
            bos_token_id=self.tokenizer.vocab["<|begin_of_text|>"],
            pad_token_id=self.tokenizer.vocab["<pad>"],
            modality = self.modality
        )[0]
    
        # SHAP explainer setup
        background = np.zeros((1, p), dtype=np.float32)  # All features removed
        x_explain = np.ones((1, p), dtype=np.float32)    # All features present
    
        def shap_model(masks):
            return self.shap_wrapper(
                masks,
                audio_tokens,
                video_tokens,
                full_baseline_emb,
                self.llm,
                baseline_ids,
            )
        
        if shap_alg == "sampling":
            
            explainer = shap.SamplingExplainer(
                model=shap_model,
                data=background
            )
    
            shap_values = explainer.shap_values(
                x_explain,
                nsamples=nsamples
            )
            
        elif shap_alg == "permutation":
            
            from shap.maskers import Independent
            masker = Independent(background, max_samples=100)
            
            explainer = shap.PermutationExplainer(
                model=shap_model,
                masker=masker,
                algorithm='auto'
            )
            shap_obj = explainer(x_explain, max_evals=nsamples, silent=False)
            shap_values = shap_obj.values
            
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[0]
    
        # CRITICAL: SHAP returns (p, T) - do NOT transpose
        vals = shap_values
        
        mm_raw_abs = np.sum(np.abs(vals), axis=1)
        mm_audio_abs = mm_raw_abs[:N_a].sum()
        mm_video_abs = mm_raw_abs[N_a:].sum()
        total_abs = mm_audio_abs + mm_video_abs
        
        audio_pct_abs = mm_audio_abs / total_abs
        video_pct_abs = mm_video_abs / total_abs
        
        
        # Print results
        print(f"Audio contribution (absolute): {audio_pct_abs*100:.2f}%")
        print(f"Video contribution (absolute): {video_pct_abs*100:.2f}%")
        
        return audio_pct_abs, video_pct_abs, \
                num_audio_tokens, vals
    
    def forward(self, inputs, is_trainval= True, modality = None, test_ratio_matry_audio = None, test_ratio_matry_video = None):
        
        if is_trainval:
            output_dict = self.prepare_inputs(inputs, is_trainval, test_ratio_matry_audio = test_ratio_matry_audio, test_ratio_matry_video = test_ratio_matry_video)
            
            if self.is_task_specific:
                audio_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], self.prompt_audio.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                        labels = output_dict["labels_audio"], modality = "audio")
                video_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["video_tokens"], self.prompt_video.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                        labels = output_dict["labels_video"], modality = "video")
                audiovisual_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], output_dict["video_tokens"], self.prompt_audiovisual.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                              labels = output_dict["labels_audiovisual"], modality = "audiovisual")
            else:
                audio_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], self.prompt_audio.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                        labels = output_dict["labels_audio"])
                video_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["video_tokens"], self.prompt_video.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                        labels = output_dict["labels_video"])
                audiovisual_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], output_dict["video_tokens"], self.prompt_audiovisual.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                              labels = output_dict["labels_audiovisual"])
                
            audio_loss = audio_output.loss*self.matry_weights[0] if self.matry_weights else audio_output.loss
            video_loss = video_output.loss*self.matry_weights[1] if self.matry_weights else video_output.loss
            audiovisual_loss = audiovisual_output.loss*self.matry_weights[2] if self.matry_weights else audiovisual_output.loss
                
            return audio_loss, video_loss, audiovisual_loss
        
        else:
               
            embeddings = self.prepare_inputs(inputs, is_trainval, test_ratio_matry_audio = test_ratio_matry_audio, test_ratio_matry_video = test_ratio_matry_video) 
           
            decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|end_of_text|>"], 
                                                bos_token_id = self.tokenizer.vocab["<|begin_of_text|>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                modality = modality
                                                )
            return decoded_ids
            
    
    def prepare_inputs(self, inputs, is_trainval, test_ratio_matry_audio = None, test_ratio_matry_video = None):
        
        
        if is_trainval:
            audio_features, selected_audio_rate = self.encode_audio(inputs["audio"], max(inputs["lengths"]), is_trainval = is_trainval, test_ratio_matry_audio = test_ratio_matry_audio)
            video_features, selected_video_rate = self.encode_video(inputs["video"], is_trainval = is_trainval, test_ratio_matry_video = test_ratio_matry_video)
            
            text_embeddings = self.llm.model.embed_tokens(inputs["tokens"])
            
            ignore_count_audio = 0 
            ignore_count_video = 0
            ignore_count_audiovisual = 0
            
            ignore_count_audio += self.prompt_audio_len
            ignore_count_video += self.prompt_video_len
            ignore_count_audiovisual += self.prompt_audiovisual_len
            
            video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            video_starts =  self.llm.model.embed_tokens(video_starts)
            
            video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            video_ends = self.llm.model.embed_tokens(video_ends)
            
            video_features = self.video_proj[self.matry_map_video[selected_video_rate]](video_features)
            
            video_inputs = torch.cat([video_starts, video_features, video_ends], dim=1)
            
            ignore_count_video += video_inputs.shape[1]
            ignore_count_audiovisual += video_inputs.shape[1]
            
            audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            audio_starts =  self.llm.model.embed_tokens(audio_starts)
            
            audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            audio_ends = self.llm.model.embed_tokens(audio_ends)
            
            audio_features = self.audio_proj[self.matry_map_audio[selected_audio_rate]](audio_features)
        
            audio_inputs = torch.cat([audio_starts, audio_features, audio_ends], dim=1)
            
            ignore_count_audio += audio_inputs.shape[1]
            ignore_count_audiovisual += audio_inputs.shape[1]
            
            labels_audio = torch.tensor([IGNORE_INDEX]*ignore_count_audio, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            labels_video = torch.tensor([IGNORE_INDEX]*ignore_count_video, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            labels_audiovisual = torch.tensor([IGNORE_INDEX]*ignore_count_audiovisual, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            
            labels_audio = torch.cat(
                [inputs["labels"][:, 0].unsqueeze(1), labels_audio, inputs["labels"][:, 1:]], dim=1)
            labels_video = torch.cat(
                [inputs["labels"][:, 0].unsqueeze(1), labels_video, inputs["labels"][:, 1:]], dim=1)
            labels_audiovisual = torch.cat(
                [inputs["labels"][:, 0].unsqueeze(1), labels_audiovisual, inputs["labels"][:, 1:]], dim=1)
            
            return {"text_embeddings": text_embeddings,
                    "audio_tokens": audio_inputs,
                    "video_tokens": video_inputs,
                    "labels_audio": labels_audio,
                    "labels_video": labels_video,
                    "labels_audiovisual": labels_audiovisual
                    }
            
        else:
            audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"]), is_trainval = is_trainval, test_ratio_matry_audio = test_ratio_matry_audio) if self.modality in ["audio", "audiovisual"] else None
            video_features = self.encode_video(inputs["video"], is_trainval = is_trainval, test_ratio_matry_video = test_ratio_matry_video) if self.modality in ["video", "audiovisual"] else None
            
            text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
            
            if self.modality == "audio":
                prompt_ids = self.prompt_audio
            elif self.modality == "video":
                prompt_ids = self.prompt_video
            else:
                prompt_ids = self.prompt_audiovisual#.device)
        
            text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_ids], dim=1)
        
            if video_features is not None:
                video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_starts =  self.llm.model.embed_tokens(video_starts)
                
                video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_ends = self.llm.model.embed_tokens(video_ends)
                
                
                video_features = self.video_proj[self.matry_map_video[test_ratio_matry_video]](video_features)
                
                video_inputs = torch.cat([video_starts, video_features, video_ends], dim=1)
               
                text_embeddings = torch.cat(
                    [text_embeddings[:, 0, :].unsqueeze(1), video_inputs, text_embeddings[:, 1:, :]], dim=1)
                    
            
            if audio_features is not None:
                audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_starts =  self.llm.model.embed_tokens(audio_starts)
                
                audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_ends = self.llm.model.embed_tokens(audio_ends)
                
                audio_features = self.audio_proj[self.matry_map_audio[test_ratio_matry_audio]](audio_features)
            
                audio_inputs = torch.cat([audio_starts, audio_features, audio_ends], dim=1)
                
                text_embeddings = torch.cat(
                    [text_embeddings[:, 0, :].unsqueeze(1), audio_inputs,text_embeddings[:, 1:, :] ], dim=1)
            
            
            return text_embeddings
        
    
    def encode_video(self, videos, is_trainval = None, test_ratio_matry_video = None):
        
        video_enc, _, encoder_layers = self.video_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': None})
        
        if is_trainval:
            if test_ratio_matry_video:
                video_enc = video_enc.transpose(1,2).contiguous()
                video_enc = self.avg_pool_video[self.matry_map_video[test_ratio_matry_video]](video_enc)
                video_enc = video_enc.transpose(1,2).contiguous()
                return video_enc, test_ratio_matry_video
            else:
                current_ratio = random.choice(self.downsample_ratio_video) # We choose a random rate.
                video_enc = video_enc.transpose(1,2).contiguous()
                video_enc = self.avg_pool_video[self.matry_map_video[current_ratio]](video_enc)
                video_enc = video_enc.transpose(1,2).contiguous()
                return video_enc, current_ratio
        else:
             video_enc = video_enc.transpose(1,2).contiguous()
             video_enc = self.avg_pool_video[self.matry_map_video[test_ratio_matry_video]](video_enc)
             video_enc = video_enc.transpose(1,2).contiguous()
             return video_enc
        
    
    def encode_audio(self, audio, max_len, is_trainval = None, test_ratio_matry_audio = None, return_num_tokens= False):
            
        #if is_trainval: # In test time we don't have to convert to float32 and then convert back to bfloat16!
        audios = audio.to(torch.float32)
        audios = audios.cpu().numpy()
        audio_extract = self.audio_frontend(audios.squeeze(-1), return_tensors="pt",sampling_rate =16000).input_features
        audio_enc = self.audio_encoder(audio_extract.cuda().to(torch.bfloat16)).last_hidden_state
    
        # Due to the 30s padding required by Whisper, we drop the tokens that correspond to the padded 0s. As 1s corresponds to 50 tokens, we truncate acccordingly.
        audio_enc = audio_enc[:, 0: max(int(max_len/16000*50), 25) , :]
        num_audio_tokens = audio_enc.shape[1]
    
        if is_trainval:
            if test_ratio_matry_audio:
                audio_enc = audio_enc.transpose(1,2).contiguous()
                audio_enc = self.avg_pool_audio[self.matry_map_audio[test_ratio_matry_audio]](audio_enc) 
                audio_enc = audio_enc.transpose(1,2).contiguous()
                return audio_enc, test_ratio_matry_audio
            else:
                current_ratio = random.choice(self.downsample_ratio_audio) # We choose a random rate.
                audio_enc = audio_enc.transpose(1,2).contiguous()
                audio_enc = self.avg_pool_audio[self.matry_map_audio[current_ratio]](audio_enc)
                audio_enc = audio_enc.transpose(1,2).contiguous()
                return audio_enc, current_ratio
        else:
            audio_enc = audio_enc.transpose(1,2).contiguous()
            audio_enc = self.avg_pool_audio[self.matry_map_audio[test_ratio_matry_audio]](audio_enc) 
            audio_enc = audio_enc.transpose(1,2).contiguous()
            return (audio_enc, num_audio_tokens) if return_num_tokens else audio_enc