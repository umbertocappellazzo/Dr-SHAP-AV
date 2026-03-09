#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:54:16 2026

@author: umbertocappellazzo
"""

import logging
from argparse import ArgumentParser

from datamodule.data_module import DataModule_LLM
from scripts.lightning_OmniAVSR_shap import ModelModule_LLM

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

def get_trainer(args):
    return Trainer(precision='bf16-true',
                   num_nodes=1,
                   devices=1,
                   accelerator="gpu",
                   logger=WandbLogger(name=args.exp_name, project=args.wandb_project)
                   )
 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-name",
        default= None,
        type=str,
        help="Experiment name.",
    )
    parser.add_argument(
        "--wandb-project",
        default= None, 
        type=str,
        help="wandb project name where to track the experiment metrics.",
    )
    parser.add_argument(
        "--modality",
        default="video",
        type=str,
        help="Type of input modality.",
        choices=["audio", "video", "audiovisual"],
    )
    parser.add_argument(
        "--pretrained-model-path",                      
        default= None,
        type=str,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--root-dir",
        default= None,
        type=str,
        help="Root directory of preprocessed dataset.",
    )
    parser.add_argument(
        "--is-task-specific",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--use-shared-lora-task-specific",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--is-single-matry-projector",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--matry-weights",
        nargs="*",
        default=None, 
        type=float,
        help="Weights to apply to ASR, VSR, and AVSR tasks. If None, all weights are set to 1.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        type=str,
        help="Filename of testing label list.",
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-video-path",
        default= None, 
        type=str,                                                               
    )
    parser.add_argument(
        "--use-lora-avhubert",
        default = False,
        type = bool,
        help= "Whether to apply LoRA to the transformer module of AV-HuBERT."
        )
    parser.add_argument(
        "--llm-model",
        default= None,
        type=str,
        help="LLM model name",
    )
    parser.add_argument(
        "--audio-encoder-name",
        default = None,
        type = str
    )
    parser.add_argument(
        "--intermediate-size",
        default= 2048,
        type=int,
        help="Intermediate size of the projector.",
    )
    parser.add_argument(
        "--prompt-audio",
        default= "Transcribe speech to text.",
        type=str,
        help="The audio prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-video",
        default= "Transcribe video to text.",
        type=str,
        help="The visual prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-audiovisual",
        default= "Transcribe speech and video to text.",
        type=str,
        help="The audiovisual prompt for the LLM.",
    )
    parser.add_argument(
        "--unfrozen-modules",
        nargs="*",
        default= [None],
        help="Which modules to train."
    )
    parser.add_argument(
        "--add-PEFT-LLM",
        default= None,
        type= str,
        help="Whether to add a PEFT module to the LLM.",
        choices= [None, "lora"]
    )
    parser.add_argument(
        "--rank",
        default= 64,
        type=int,
        help="Rank for LoRA."
    )
    parser.add_argument(
        "--alpha",
        default= 8,
        type=int,
        help="Alpha for LoRA."
    )
    parser.add_argument(
        "--downsample-ratio-audio",
        nargs="*",
        default=3,
        type=int,
        help="Downsample audio ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        nargs="*",
        default=3,
        type=int,
        help="Downsample video ratio.",
    )
    parser.add_argument(
        "--test-specific-ratio",
        default = False,
        type = bool,
        help= "Whether to test Omni-AVSR on a specific audio and video compresison rate."
        )
    parser.add_argument(
        "--test-specific-modality",
        default = False,
        type = bool,
        help= "Whether to test Omni-AVSR on a specific task."
        )
    parser.add_argument(
        "--downsample-ratio-test-matry-audio",
        default=None,
        type=int,
        help="Downsample audio ratio for eval.",
    )
    parser.add_argument(
        "--downsample-ratio-test-matry-video",
        default=None,
        type=int,
        help="Downsample visual ratio for eval.",
    )
    parser.add_argument(
        "--task-to-test",
        default=None,
        type=str,
        choices = ["audio", "video", "audiovisual"],
        help="Task to evaluate Omni-AVSR on.",
    )
    parser.add_argument(
        "--max-dec-tokens",
        default= 32,
        type=int,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--num-beams",
        default= 15,
        type=int,
        help="Beams used for beam search.",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set.",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default= 999999,  
        help="Level of signal-to-noise ratio (SNR).",
    )
    parser.add_argument(
        "--noise-type",
        default = "babble",
        type = str,
        choices=["babble", "music", "speech", "sound"],
        help="The noise type to use during evaluation."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )

    # SHAP-related arguments.

    parser.add_argument(
        "--compute-shap",
        default=False,
        type=bool,
        help="Whether to compute SHAP values."
    )
    parser.add_argument(
        "--num-samples-shap",
        default=2000,
        type=int,
        help="Number of coalitions to use.",
    )
    parser.add_argument(
        "--shap-alg",
        default="permutation",
        type=str,
       choices=["permutation", "sampling"],
    )
    parser.add_argument(
        "--output-path-shap",
        default=None,
        type=str,
        help="Directory to save the output data for shap additional analyses",
    )
    
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")

def cli_main():
    args = parse_args()
    init_logger(args.debug)
    
    modelmodule = ModelModule_LLM(args)
    datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)

    if args.test_specific_ratio or args.test_specific_modality:
        if args.test_specific_ratio and args.test_specific_modality:
            args.modality = args.task_to_test
            trainer.test(model=modelmodule, datamodule=datamodule)
        elif args.test_specific_ratio:
            print("Evaluating on the ASR task!")
            args.modality = "audio"
            trainer.test(model=modelmodule, datamodule=datamodule)

            print("Evaluating on the VSR task!")
            args.modality = "video"
            trainer.test(model=modelmodule, datamodule=datamodule)

            print("Evaluating on the AVSR task!")
            args.modality = "audiovisual"
            trainer.test(model=modelmodule, datamodule=datamodule)
        else:
            args.modality = args.task_to_test

            if args.modality == "audio":
                for rate_audio in args.downsample_ratio_audio:
                    args.downsample_ratio_test_matry_audio = rate_audio
                    trainer.test(model=modelmodule, datamodule=datamodule)
            elif args.modality == "video":
                for rate_video in args.downsample_ratio_video:
                    args.downsample_ratio_test_matry_video = rate_video
                    trainer.test(model=modelmodule, datamodule=datamodule)
            else:
                for rate_video in args.downsample_ratio_video:
                    args.downsample_ratio_test_matry_video = rate_video
                    for rate_audio in args.downsample_ratio_audio:
                        args.downsample_ratio_test_matry_audio = rate_audio
                        trainer.test(model=modelmodule, datamodule=datamodule)


    else: # We evaluate for each task and for each compression ratio.

        print("Evaluating on the ASR task!")
        args.modality = "audio"
        for rate_audio in args.downsample_ratio_audio:
            args.downsample_ratio_test_matry_audio = rate_audio
            trainer.test(model=modelmodule, datamodule=datamodule)

        print("Evaluating on the VSR task!")
        args.modality = "video"
        for rate_video in args.downsample_ratio_video:
            args.downsample_ratio_test_matry_video = rate_video
            trainer.test(model=modelmodule, datamodule=datamodule)

        print("Evaluating on the AVSR task!")
        args.modality = "audiovisual"
        for rate_video in args.downsample_ratio_video:
            args.downsample_ratio_test_matry_video = rate_video
            for rate_audio in args.downsample_ratio_audio:
                args.downsample_ratio_test_matry_audio = rate_audio
                trainer.test(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    cli_main()