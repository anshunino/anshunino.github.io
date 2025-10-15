---
layout: page
title: "deltaMic"
description: ""
img: 'assets/img/dmic_pipeline.jpg'
date: 2025-06-25 14:00:00 +0200
importance: 1
category: research
---

<link rel="stylesheet" href="{{ '/assets/css/research.css' | relative_url }}">

<div class="hero">
  <h2>Inverse 3D Microscopy Rendering for Embryo Shape Inference with Active Mesh</h2>

  <p class="authors">
    <strong>Sacha Ichbiah¹, Anshuman Sinha¹², Fabrice Delbary¹, Hervé Turlier¹</strong><br>
    ¹ Collège de France, CNRS, Inserm, PSL University, Paris, France<br>
    ² Université Paris Cité
  </p>
</div>

---

## Overview

**deltaMic** is a CUDA-based *differentiable 3D renderer* for fluorescence microscopy.  
It models image formation as a **Fourier-space convolution** between the microscope’s point spread function (PSF) and a **triangular mesh representation** of the specimen.

By jointly optimizing **mesh geometry** and **optical parameters** directly from raw images, deltaMic achieves robust 3D reconstruction **without labeled training data or priors**.

{% include figure.html path="assets/img/dmic_pipeline.jpg" title="Figure 1 – deltaMic pipeline overview." class="centered-img" %}

---

## Methodology

Microscopy rendering is modeled as:

$$
I(x) = (u_\Lambda * h)(x) = \int u_\Lambda(p)\, h(x-p)\, d^3p,
$$

and equivalently in Fourier space:

$$
I(x) = \mathcal{F}^{-1}\left[\hat{u}_\Lambda \cdot \hat{h}\right](x),
$$

reducing computational complexity from O(n^6) to O(n^3 . log(n^3)).

deltaMic begins from an initial mesh and iteratively updates both the **vertex coordinates** and **PSF parameters** to minimize the weighted image difference between rendered and experimental microscopy data.

![Active Mesh Optimization](assets/img/active_mesh_optimization.gif){: .centered-img }
*Figure 2 – Active mesh optimization over successive iterations.*

---

## Results Across Species

### Embryo Shape Inference

deltaMic accurately reconstructs early-stage embryo morphologies across species and imaging modalities.

{% include figure.html path="assets/img/dmic_fig10.png" title="Figure 3 – Inferred cellular geometries for ascidian, mouse, and C. elegans embryos." class="centered-img" %}

{% include video.html path="assets/video/Supplementary_mov_s8.mp4" autoplay=true loop=true muted=true playsinline=true class="centered-video" %}
*Supplementary Video – Shape inference and synthetic rendering convergence.*

---

## Benchmarking

{::options parse_block_html="true" /}
<details open class="benchmark-section">
  <summary><strong>Benchmarking Results</strong></summary>

  <p>To assess generality, deltaMic was compared against <strong>DM3D</strong>, a leading active-mesh segmentation framework implemented as a Fiji plug-in.</p>

  {% include figure.html path="assets/img/dmic_fig9.png" title="Figure 4 – Comparison between DM3D and deltaMic on 3D mouse organoid data." class="centered-img" %}

  <blockquote>
  deltaMic demonstrates improved reconstruction of multicellular membrane structures while maintaining photometric fidelity.  
  Performance on nucleus-only datasets is limited due to design assumptions targeting multicellular surfaces.
  </blockquote>

</details>
{::options parse_block_html="false" /}

---

## Discussion

deltaMic bridges *physics-based rendering* and *inverse modeling* in biological imaging.  
This approach enables quantitative analysis of morphogenesis by linking observed fluorescence to underlying 3D geometry and optical parameters.

**Current limitations**
- Dependence on an adequate initial mesh  
- Computational load for large volumetric datasets  

**Ongoing directions**
- Automated mesh initialization  
- Temporal morphodynamic inference (tension, curvature, pressure)  
- Integration with biophysical simulation pipelines  

---

## References & Resources

- **Paper:** [arXiv:2303.10440](https://doi.org/10.48550/arXiv.2303.10440)  
  *(ICCV 2025 version forthcoming)*  
- **Code:** [GitHub Repository](https://github.com/VirtualEmbryo/deltaMic)  
- **Contact:** [anshuman.sinha@etu.u-paris.fr](mailto:anshuman.sinha@etu.u-paris.fr)

```bibtex
@misc{ichbiah2025inverse3dmicroscopyrendering,
  title={Inverse 3D Microscopy Rendering for Cell Shape Inference with Active Mesh},
  author={Sacha Ichbiah and Anshuman Sinha and Fabrice Delbary and Hervé Turlier},
  year={2025},
  eprint={2303.10440},
  archivePrefix={arXiv},
  primaryClass={physics.bio-ph},
  url={https://arxiv.org/abs/2303.10440}
}
