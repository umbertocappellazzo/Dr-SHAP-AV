#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:09:38 2024

@author: umbertocappellazzo
"""
import sys
sys.path.append("..")

import numpy as np
import shap
import torch.nn.functional as F
import torch
from torch import nn
from .Llama_LoRA import LlamaForCausalLM_lora
from transformers import WhisperModel, LlamaForCausalLM, AutoFeatureExtractor
import fairseq
from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq, AVHubertSeq2SeqConfig
from av_hubert.avhubert.hubert_lora import AVHubertModel_lora
import math

#from av_hubert.avhubert.hubert import AVHubertModel
#from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq
#from AV_HuBERT_encoder import avhubertConfig, avhubert_encoder


class Top_K_MoE(nn.Module):
    def __init__(self, num_experts, top_k, in_dim, intermediate_dim, llm_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, num_experts, bias=False)
        
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, intermediate_dim), nn.ReLU(), nn.Linear(intermediate_dim, llm_dim), nn.LayerNorm(llm_dim)) for _ in range(num_experts)])
        self.top_k = top_k
        self.num_experts = num_experts
        self.llm_dim = llm_dim
        
    def forward(self, hidden_states, print_experts_analysis = False):
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.llm_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.llm_dim)
        
        if print_experts_analysis:
            return final_hidden_states, router_logits, selected_experts
        
        return final_hidden_states, router_logits
            
# Note that the computation of this load balancing loss is applied to Mixture of Projectors, so it is not applied on a per-layer basis as
# in the original code in transformers/src/transformers/models/mixtral/modeling_mixtral.py. So we apply a few changes.

def router_z_loss_func(router_logits):
        router_z_loss = torch.logsumexp(router_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()
        
        return router_z_loss

def load_balancing_loss_func(gate_logits, num_experts, top_k):
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """

    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

IGNORE_INDEX = -100

class AVSR_LLMs(nn.Module):
    def __init__(self, modality, pretrain_avhubert_enc_video,  use_lora_avhubert, llm_model, 
                 hidden_size, intermediate_size, tokenizer, prompt, pad_id, 
                 downsample_ratio_audio, downsample_ratio_video, audio_encoder_name,
                 unfrozen_modules, max_dec_tokens, num_beams, PEFT_LLM_name = None, peft_config_llm = None,
                 MoP_config = None
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
        self.MoP_config = MoP_config
            
        if modality == "audio" or modality == "audiovisual":
           
            print("Instantiating whisper!")    
            self.audio_encoder = WhisperModel.from_pretrained(self.audio_encoder_name).encoder
            self.audio_frontend = AutoFeatureExtractor.from_pretrained(self.audio_encoder_name)
            self.audio_encoder.requires_grad_(False)
            self.audio_encoder.train() # This must be explicitly done as by default the from_pretrained HF models are in eval mode when initialized (this is the opposite for pytorch!)--> cause a break in deepspeed 3! https://github.com/Lightning-AI/pytorch-lightning/issues/19467
            
            self.avg_pool_audio = nn.AvgPool1d(self.downsample_ratio_audio)
            
            
        if modality == "video" or modality == "audiovisual":
            if pretrain_avhubert_enc_video:
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
                
                self.avg_pool_video = nn.AvgPool1d(self.downsample_ratio_video)
        
        self.audiovisual_proj = Top_K_MoE(MoP_config.MoP_experts, MoP_config.top_k, video_dim, intermediate_size, hidden_size)
        
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
        
        self.prompt = prompt
        
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
        
        if "lora_avhubert" in unfrozen_modules:
            
            if self.modality == "video": 
                print("Unfreezing LoRA for AV-HuBERT video encoder!")
                for block_idx in range(24):
                    # If we don't use the correct config parameters --> my initial implementation.
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
            output_hidden_states=False
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
    
    def forward_shap(self, inputs, nsamples, shap_alg):
        """
        Compute SHAP values for audio and video contributions to LLM-based AVSR.
        Uses "remove from full" approach: mask=1 means keep, mask=0 means remove.

        """
        
        # Encode audio/video
        
        audio_features, num_audio_tokens = self.encode_audio(inputs["audio"], max(inputs["lengths"]), return_num_tokens=True) if self.modality in ["audio", "audiovisual"] else None
        video_features = self.encode_video(inputs["video"]) if self.modality in ["video", "audiovisual"] else None
        
    
        device = inputs["tokens"].device
    
        N_a = audio_features.shape[1] if audio_features is not None else 0
        N_v = video_features.shape[1] if video_features is not None else 0
        p = N_a + N_v
    
        # Build prompt embeddings
        text_embeddings_raw = self.llm.model.embed_tokens(inputs["tokens"])
    
        prompt_tokens_start_at = 1
        prompt_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids[
            :, prompt_tokens_start_at:-1
        ].to(device)
    
        prompt_embeddings = self.llm.model.embed_tokens(
            prompt_ids.expand(inputs["tokens"].shape[0], -1)
        )
    
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
        audiovisual_features = torch.cat((audio_features, video_features), dim = 1)
        
        audiovisual_features, _ = self.audiovisual_proj(audiovisual_features)
        
        audio_tokens = audiovisual_features[:, :N_a]
        video_tokens = audiovisual_features[:, N_a:]
        
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
        
    
    def forward(self, inputs, is_trainval= True):
        
        if is_trainval:
            
            embeddings, labels, load_balancing_loss, router_z_loss = self.prepare_inputs(inputs, is_trainval)
            
            outputs = self.llm(inputs_embeds = embeddings, labels = labels)
            return outputs[0], load_balancing_loss, router_z_loss
                
        
        else:
            embeddings = self.prepare_inputs(inputs, is_trainval)
            
            decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|end_of_text|>"], 
                                                bos_token_id = self.tokenizer.vocab["<|begin_of_text|>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                )
            return decoded_ids
            
    
    def prepare_inputs(self, inputs, is_trainval):

        audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"])) if self.modality in ["audio", "audiovisual"] else None
        video_features = self.encode_video(inputs["video"]) if self.modality in ["video", "audiovisual"] else None
        
        text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
        
        ignore_count = 0 
        
        # An important note here: the tokenizer by default inserts the EOS and BOS tokens. Since we do that already in the collate_LLM, here we need to
        # get rid of them explicitly --> [:,1:-1].
       
        prompt_tokens_start_at = 1
        prompt_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1].to(text_embeddings_.device)
        prompt_embeddings = self.llm.model.embed_tokens(prompt_ids.expand(inputs["tokens"].shape[0],-1))
        
        if is_trainval:
            text_embeddings = torch.cat(
                [torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1), text_embeddings_[:, 1:, :]], 
                dim=1)
        else:
            text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1)
        
        ignore_count += prompt_embeddings.shape[1]
        
        
        video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
        video_starts =  self.llm.model.embed_tokens(video_starts)
        
        video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
        video_ends = self.llm.model.embed_tokens(video_ends)
        
        audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
        audio_starts =  self.llm.model.embed_tokens(audio_starts)
        
        audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
        audio_ends = self.llm.model.embed_tokens(audio_ends)
        
        audiovisual_features = torch.cat((audio_features, video_features), dim = 1)
        
        ignore_count += audiovisual_features.shape[1] + 4
        
        audiovisual_features, routing_logits_audiovisual = self.audiovisual_proj(audiovisual_features)
        
        audio_len = audio_features.shape[1]
        
        text_embeddings = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), audio_starts, audiovisual_features[:, :audio_len, :], audio_ends, video_starts, audiovisual_features[:,audio_len:, :], video_ends, text_embeddings[:, 1:, :]), dim = 1)
        

        if inputs["labels"] is not None:
            labels = torch.tensor([IGNORE_INDEX]*ignore_count, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            labels = torch.cat(
                [torch.cat([inputs["labels"][:, 0].unsqueeze(1), labels], dim=1), inputs["labels"][:, 1:]], 
                dim=1)
        else:
            labels = None
        
        if is_trainval:
            return text_embeddings, labels, load_balancing_loss_func(routing_logits_audiovisual, self.MoP_config.MoP_experts, self.MoP_config.top_k), router_z_loss_func(routing_logits_audiovisual)
        else:
            return text_embeddings
        
    
    def encode_video(self, videos):
        
        video_enc, _, encoder_layers = self.video_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': None})
        
        if self.downsample_ratio_video != 1:
            video_enc = video_enc.transpose(1,2).contiguous()
            video_enc = self.avg_pool_video(video_enc)
            video_enc = video_enc.transpose(1,2).contiguous()
            
        return video_enc
    
    def encode_audio(self, audio, max_len, return_num_tokens= False):
            
        #if is_trainval: # In test time we don't have to convert to float32 and then convert back to bfloat16!
        audios = audio.to(torch.float32)
        audios = audios.cpu().numpy()
        
        audio_extract = self.audio_frontend(audios.squeeze(-1), return_tensors="pt",sampling_rate =16000).input_features
       
        audio_enc = self.audio_encoder(audio_extract.cuda().to(torch.bfloat16)).last_hidden_state
    
        # Due to the 30s padding required by Whisper, we drop the tokens that correspond to the padded 0s. As 1s corresponds to 50 tokens, we truncate acccordingly.
        audio_enc = audio_enc[:, 0: max(int(max_len/16000*50), 25) , :]
        
        num_audio_tokens = audio_enc.shape[1]
        
        if self.downsample_ratio_audio != 1:
            audio_enc = audio_enc.transpose(1,2).contiguous()
            audio_enc = self.avg_pool_audio(audio_enc)
            audio_enc = audio_enc.transpose(1,2).contiguous()
                    
        return (audio_enc, num_audio_tokens) if return_num_tokens else audio_enc