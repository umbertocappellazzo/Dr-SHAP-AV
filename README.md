# Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2024.12345-b31b1b.svg)](?)
[![Website](https://img.shields.io/badge/🌐-Website-blue.svg)](https://umbertocappellazzo.github.io/Dr-SHAP-AV/)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=umbertocappellazzo.Dr-SHAP-AV)](https://github.com/umbertocappellazzo/Dr-SHAP-AV)
[![GitHub Stars](https://img.shields.io/github/stars/umbertocappellazzo/Dr-SHAP-AV?style=social)](https://github.com/umbertocappellazzo/Dr-SHAP-AV/stargazers)

**[Umberto Cappellazzo¹](#) · [Stavros Petridis¹²](#) · [Maja Pantic¹²](#)**

¹Imperial College London ²NatWest AI Research

### 📄 [`Paper`](?) | 🌐 [`Project Page`](https://umbertocappellazzo.github.io/Dr-SHAP-AV/) | 💻 [`Code`](https://github.com/umbertocappellazzo/Dr-SHAP-AV) | 🔖 [`BibTeX`](#-citation)

</div>

---

## 📢 News
- **[03-2026]** 🚀 Code and models released!
- **[03-2026]** 📝 Paper submitted to arXiv.


---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Setup](#-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Checkpoints](#-checkpoints)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

## 🌟 Highlights

Dr. SHAP-AV is a unified Shapley-based framework for understanding how AVSR models balance audio and visual modalities across model atchitectures, decoding stages, and acoustic conditions. 

<div align="center">
  <img src="assets/SHAP_main_diagram.png" alt="Architecture" width="800"/>
  <p><i>The three proposed SHAP-based analyses in Dr. SHAP-AV: Global/Generative/Temporal Alignment SHAP.</i></p>
</div>

Through Dr. SHAP, we find multiple key findings:
- **Persistent Audio Bias**: AVSR models tend to shift toward visual reliance as noise increases, yet maintain high audio contributions
even under severe degradation.
- **Dynamic Generation Shift**: Whisper-Flamingo and Omni-AVSR progressively increase audio reliance during generation, while AV-
HuBERT maintains stable modality balance throughout.
- **Robust Temporal Alignment**: Both audio and visual modalities independently maintain temporal correspondence between input features
and output tokens, even under severe acoustic noise.
- **Noise-Type Sensitivity**: The degree of visual shift depends on noise type and severity: more challenging acoustic conditions induce
greater reliance on visual information.
- **Architecture-Dependent Duration Effect**: The relationship between utterance duration and modality balance is architecture-dependent, with no universal trend across models or conditions.
- **SNR-Driven Modality Balance**: Modality contributions are determined by acoustic conditions rather than recognition difficulty.

---

## 🛠 Setup 
This repository contains the code to reproduce the results of our paper. Specifically, we include here the three `LLM-based models`: **1)** Llama-AVSR, **2)** Llama-SMoP, **3)** Omni-AVSR. Below we point to the repositories regarding the three `cross-attention encoder-decoder architectures`. Since each of them requires an ad-hoc environment, we created three dedicated repositories. On the contrary, Llama-AVSR, Llama-SMoP, and Omni-AVSR share the same environment.

- [Auto-AVSR](https://github.com/umbertocappellazzo/auto_avsr_shap)
- [AV-HuBERT](https://github.com/umbertocappellazzo/av_hubert_shap)
- [Whisper-Flamingo](https://github.com/umbertocappellazzo/whisper-flamingo-shap)


Our setup follows that of [Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR).

## 🔖 Citation

If you find our work useful, please cite:

```bibtex
@article{cappellazzo2026ODrSHAPAV,
  title={Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition},
  author={Umberto, Cappellazzo and Stavros, Petridis and Maja, Pantic},
  journal={arXiv preprint arXiv:?},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- Our code relies on [Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR), [Omni-AVSR](https://github.com/umbertocappellazzo/Omni-AVSR), [AV-HuBERT](https://github.com/facebookresearch/av_hubert), [Whisper-Flamingo](https://github.com/roudimit/whisper-flamingo), and [Auto-avsr](https://github.com/mpc001/auto_avsr) repositories.

---

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: umbertocappellazzo@gmail.com
- Visit our [project page](https://umbertocappellazzo.github.io/Dr-SHAP-AV/) and our [preprint](https://arxiv.org/abs/?)

---


