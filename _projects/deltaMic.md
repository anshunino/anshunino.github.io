---
layout: page
title:  "deltaMic"
description: "My work during the 1st year of my PhD at the Turlier Lab"
img: 'assets/img/dmic_pipeline.jpg'
date:   2025-06-25 14:00:00 +0200
importance: 1
category : research

---

# deltaMic: Inverse 3D Microscopy Rendering for Embryo Shape Inference with Active Mesh

**Sacha Ichbiah¹, Anshuman Sinha¹², Fabrice Delbary¹, Hervé Turlier¹**  
¹ Collège de France, CNRS, Inserm, PSL University, Paris, France  
² Université Paris Cité  

---

## 🧩 Abstract

Conventional methods for biological shape inference rely on **manual annotation** or **deep learning** that requires large labeled datasets — especially hard to obtain for 3D biological structures.

**deltaMic** is a **CUDA-based differentiable 3D renderer** for fluorescence microscopy.  
It mimics the microscope’s image formation process using **spatial convolutions in Fourier-space**, coupling a **mesh-based model** with a **point spread function (PSF)**.

> By jointly optimizing both shape and microscopy parameters from raw images, deltaMic removes the need for training datasets or heuristic priors.

{% include figure.html path="assets/img/dmic_pipeline.jpg" title="Pipeline Overview" class="img-fluid rounded z-depth-1"%}
*Figure 1. Overview of the deltaMic pipeline.*

---

## ⚙️ Methodology

### Image Formation as a Convolution
Microscopy rendering is modeled as:

$$ I(x) = (u_\Lambda * h)(x) = \int u_\Lambda(p)\, h(x-p)\, d^3p $$

Equivalent in Fourier-space to:  
$$ I(x) = \mathcal{F}^{-1}[ \hat{u}_\Lambda \cdot \hat{h} ](x) $$

This reduces computational complexity from **O(n⁶)** to **O(n³ log n³)**.


---

## 🧠 Optimization Process

Starting from an initial mesh, deltaMic **optimizes**:

- 3D vertex positions  
- PSF parameters  

to minimize the weighted L2 loss between **rendered** and **real** microscopy images.

![Active Mesh Optimization](assets/img/active_mesh_optimization.gif)  
*Figure 2. Mesh optimization over iterations.*

---

## 🧫 Results Across Species

### Embryo Shape Inference

{% include figure.html path="assets/img/dmic_fig10.png" title="" class="img-fluid rounded z-depth-1"%}

{% include video.html path="assets/video/Supplementary_mov_s8.mp4" title=""%}
*Video. Shape inference and synthetic rendering convergence.*

---

## 🧪 Benchmarking

deltaMic was benchmarked against **DM3D**, a Fiji plugin for active-mesh segmentation.

{% include figure.html path="assets/img/dmic_fig9.png" title="" class="img-fluid rounded z-depth-1"%}

> deltaMic outperforms DM3D on **membrane reconstruction accuracy** and **synthetic fidelity**, though less suited for nucleus membranes.

---

## 📚 Resources  

- **Paper**: [Arxiv link](https://doi.org/10.48550/arXiv.2303.10440) , [ICCV 2025 version] #TBD
- **Code / Repository**: [GitHub link](https://github.com/VirtualEmbryo/deltaMic)
- **BibTeX**:  
  ```bibtex
  @misc{ichbiah2025inverse3dmicroscopyrendering,
      title={Inverse 3D Microscopy Rendering for Cell Shape Inference with Active Mesh}, 
      author={Sacha Ichbiah and Anshuman Sinha and Fabrice Delbary and Hervé Turlier},
      year={2025},
      eprint={2303.10440},
      archivePrefix={arXiv},
      primaryClass={physics.bio-ph},
      url={https://arxiv.org/abs/2303.10440}, 
  }
- 📧 Contact: [anshuman.sinha@etu.u-paris.fr](mailto:anshuman.sinha@etu.u-paris.fr)

---

## 🎥 Bonus

[![](assets/img/deltaMic_vid_Thumbnail.png)](https://youtu.be/55I-_FWINvI)
*Watch the short presentation of deltaMic (ICCV 2025 poster spotlight)*

---

**© 2025 Turlier Lab – Multiscale Physics of Morphogenesis, CIRB, Paris**
