
<p align="center">
    <img src="https://github.com/user-attachments/assets/fba781e5-497d-44fa-abb5-07b3b3e8a471" width="256" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="https://github.com/PKU-YuanGroup/WF-VAE/">WF-VAE: Enhancing Video VAE by Wavelet-Driven Energy Flow for Latent Video Diffusion Model</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<h5 align="center">
    
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.17459-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17459)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/PKU-YuanGroup/WF-VAE/blob/main/LICENSE)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPKU-YuanGroup%2FWF-VAE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FPKU-YuanGroup%2FWF-VAE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)
[![GitHub repo stars](https://img.shields.io/github/stars/PKU-YuanGroup/WF-VAE?style=flat&logo=github&logoColor=whitesmoke&label=Stars)](https://github.com/PKU-YuanGroup/WF-VAE/stargazers)

</h5>

<details open><summary>💡 I also have other projects that may interest you ✨. </summary><p>
<!--  may -->

> [**Open-Sora-Plan**](https://github.com/PKU-YuanGroup/Open-Sora-Plan) <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social)](https://github.com/PKU-YuanGroup/Open-Sora-Plan) <br>

</p></details>

# 📰 News


* **[2024.11.27]**  🔥🔥🔥  We have published our [report](assets/report.pdf), which provides comprehensive training details and includes additional experiments. 
* **[2024.11.25]**  🔥🔥🔥 We have released our 16-channel WF-VAE-L model along with the training code.  Welcome to download it from [Huggingface](https://huggingface.co/chestnutlzj/WF-VAE-L-16Chn).

# 😮 Highlights

WF-VAE utilizes a multi-level wavelet transform to construct an efficient energy pathway, enabling low-frequency information from video data to flow into latent representation. This method achieves competitive reconstruction performance while markedly reducing computational costs.

### 💡 Simpler Architecture, Faster Encoding

- This architecture substantially improves speed and reduces training costs in large-scale video generation models and data processing workflows.

### 🔥 Competitive Reconstruction Performance with SOTA VAEs

- Our experiments demonstrate competitive performance of our model against SOTA VAEs.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e14cfd31-c5c1-4b34-af60-5a5fc2071483" style="max-width: 80%;">
</div>

# 🚀 Main Results

## Reconstruction

<div align="center">
  <img src="https://github.com/user-attachments/assets/0b9d6203-ea31-47b0-86b6-fbfaf96ddb37" style="max-width: 80%;">
</div>


<table>
  <thead>
    <tr>
      <th>WF-VAE</th>
      <th>CogVideoX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img src="https://github.com/user-attachments/assets/da74cce6-7878-4aff-ba4a-ed2b3c23f530" alt="WF-VAE">
      </td>
      <td>
        <img src="https://github.com/user-attachments/assets/a7c8c5f4-8487-485b-80d0-81caf2b01d9f" alt="CogVideoX">
      </td>
    </tr>
  </tbody>
</table>


## Efficiency

We conduct efficiency tests at 33-frame videos using float32 precision on an H100 GPU. All models operated without block-wise inference strategies. Our model demonstrated performance comparable to state-of-the-art VAEs while **significantly reducing encoding costs**.

<div align="center">
  <img src="https://github.com/user-attachments/assets/53f74160-81f0-486e-b294-10dbb5bed8e5" style="max-width: 80%;">
</div>

# 🛠️ Requirements and Installation

```bash
git clone https://github.com/PKU-YuanGroup/WF-VAE
cd WF-VAE
conda create -n wfvae python=3.10 -y
conda activate wfvae
pip install -r requirements.txt
```

# 🤖 Reconstructing Video or Image

To reconstruct a video or an image, execute the following commands:

## Video Reconstruction

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/recon_single_video.py \
    --model_name WFVAE \
    --from_pretrained "Your VAE" \
    --video_path "Video Path" \
    --rec_path rec.mp4 \
    --device cuda \
    --sample_rate 1 \
    --num_frames 65 \
    --height 512 \
    --width 512 \
    --fps 30 \
    --enable_tiling
```

## Image Reconstruction

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/recon_single_image.py \
    --model_name WFVAE \
    --from_pretrained "Your VAE" \
    --image_path assets/gt_5544.jpg \
    --rec_path rec.jpg \
    --device cuda \
    --short_size 512 
```

For further guidance, refer to the example scripts: `examples/rec_single_video.sh` and `examples/rec_single_image.sh`.

# 🗝️ Training & Validating

The training & validating instruction is in [TRAIN_AND_VALIDATE.md](TRAIN_AND_VALIDATE.md).

# 👍 Acknowledgement

- Open-Sora Plan - https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Allegro - https://github.com/rhymes-ai/Allegro
- CogVideoX - https://github.com/THUDM/CogVideo
- Stable Diffusion - https://github.com/CompVis/stable-diffusion

# 🔒 License

This project is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
