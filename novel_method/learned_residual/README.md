# Learned Transmission Residual Module (`novel_method/learned_residual`)

## Purpose
An ultra-lightweight, CPU-compatible neural module that predicts a bounded scalar transmission residual:
$$\Delta t(x, y) = 0.30 \cdot \tanh(\text{TinyUNet}(\mathbf{X})(x, y)) \in [-0.30, +0.30]$$
augmenting the physics-based spatial fusion estimate $t_{\text{fused}}(x, y)$.

## Architectural Specifications
- **Trainable Parameters:** Exactly 13,921
- **Input Dimensions:** 8 channels at guide resolution ($240 \times 135$):
  - `0–2`: Normalized RGB guide image
  - `3`: $t_{\text{fused}}$ (Color Attenuation Prior & Dark Channel Prior weighted fusion)
  - `4`: $t_{\text{cap}}$ (Color Attenuation Prior transmission)
  - `5`: $t_{\text{dcp}}$ (Dark Channel Prior transmission)
  - `6`: $w_{\text{dcp}}$ (Spatially adaptive density fusion weight)
  - `7`: Atmospheric veil proxy $\max(I_{\text{gray}} - A, 0)$
- **Output:** Scalar field $\Delta t(x, y)$ in $[-0.30, +0.30]$
- **Hardware Target:** Pure CPU execution ($> 60\text{ FPS}$ forward pass on CPU)
- **OpenCV Dependency:** STRICTLY ZERO (`cv2` forbidden)

## Physical Non-Bypass Guarantee
The learned candidate transmission:
$$t_{\text{candidate}} = \text{clip}(t_{\text{fused}} + \Delta t, 0.05, 1.00)$$
is strictly bounded by the authoritative adaptive safety floor:
$$t_{\text{constrained}} = \max\left(t_{\text{candidate}}, t_{\min}(x, y)\right)$$
ensuring that the CNN can never bypass the physics-based black-crush prevention and sky-protection floors.
