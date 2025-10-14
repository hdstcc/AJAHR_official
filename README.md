<div align="center">

# AJAHR: Amputated Joint Aware 3D Human Mesh Recovery

#### International Conference on Computer Vision (ICCV 2025)

<p>
  <strong><a href="https://chojinie.github.io/categories/">Hyunjin Cho</a><sup>*</sup></strong>, 
  <strong>Giyun Choi<sup>*</sup></strong>, 
  Jongwon Choi<sup>†</sup>
</p>

<p>
  <sup>*</sup> Equal contribution &nbsp;&nbsp;&nbsp;
  <sup>†</sup> Corresponding author
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2507.20091-brightgreen.svg)](https://arxiv.org/abs/2509.19939) [![project page](https://img.shields.io/badge/website-up-blue.svg)](https://chojinie.github.io/project_AJAHR/)

</div>

---


## 🖼️ Teaser

![Teaser](./fig/fig1.png)

## Key Idea

![Key Idea](./fig/A3D.png)

Existing SMPL-based 3D human mesh recovery approaches are fundamentally designed under the assumption that **all joints of the human body are present**. However, in amputated human bodies, **the joint existence itself is altered**, and conventional models tend to either **hallucinate non-existent joints** or enforce incorrect joint priors, resulting in **severe regression instability**.

To address this limitation, we introduce **A3D (Amputee 3D Dataset)**:

* Extends the SMPL shape space by **incorporating real-world amputee body configurations**
* Applies **joint removal and visibility-aware masking strategies** per anatomical amputation region
* Utilizes the **AJAHR Index (12 anatomical regions)** to represent each amputated area **along with its hierarchical SMPL joint descendants**
ecific distribution (P_\text{amputee})** into the training space, rather than relying solely on (P_\text{human}) from COCO/H36M

A3D is constructed with both real-world backgrounds and studio-rendered environments, featuring mesh overlays, skin tone variation across different ethnicities, and diverse clothing textures, making it suitable for robust amputation-aware mesh regression under domain variability.

A3D enables models to **learn the possibility that a joint may not exist**, shifting the problem from simple data augmentation to a **redefinition of structural priors**. This fundamentally differentiates A3D from conventional datasets such as Human3.6M and COCO, which assume intact human anatomy.

## Dataset: AJAHR Index & SMPL Mapping

: AJAHR Index & SMPL Mapping

![SMPL\_Index\_Visualization](./fig/smpl_index.png)

To support amputated-joint aware mesh reconstruction research, we release **AJAHR-Index**, a joint-group annotation protocol aligned with the SMPL kinematic hierarchy.

```
ajahr_index : {
    0 : Right Hand, 1 : Right Elbow, 2 : Right Shoulder,
    3 : Left Hand, 4 : Left Elbow, 5 : Left Shoulder,
    6 : Left Foot, 7 : Left Knee, 8 : Left Hip,
    9 : Right Foot, 10: Right Knee, 11: Right Hip
}

smpl_index : { 
    21, 23 : Right Hand,      19, 21, 23 : Right Elbow,      17, 19, 21, 23 : Right Shoulder,
    20, 22 : Left Hand,       18, 20, 22: Left Elbow,        16, 18, 20, 22 : Left Shoulder,
    7, 10 : Left Foot,        4, 7, 10 : Left Knee,          1, 4, 7, 10 : Left Hip,
    8, 11 : Right Foot,       5, 8, 11 : Right Knee,         2, 5, 8, 11 : Right Hip
}
```

> **Note:** `ajahr_index` indicates an amputation region. When an index is marked as amputated, **the corresponding region and all of its descendant joint nodes** are treated as missing. `smpl_index` refers to the **actual SMPL pose joint indices** mapped to each anatomical region.

### Label Extraction Policy (Automatic from `imgname`)

AJAHR does **not** store explicit class labels inside annotation files. Instead, **each sample's amputation level is inferred directly from its `imgname` pattern**, following the logic below:
    
```python
if 'imgname' in self.data:
    self.labels = []
    for file_name in self.imgname:
        amp_number = extract_amp_number(file_name)
        if amp_number is not None:
            self.labels.append(amp_number)
        else:
            self.labels.append(12)
    self.labels = np.array(self.labels)
else:
    #for non-amputee datasets
    self.labels = np.full((self.scale.shape[0],), 12)
```

* `amp_number ∈ {0 ~ 11}` → matches **AJAHR Index** (see mapping above)
* `12` is reserved for the **Non-amputee (default) class**
* Example filename: `subject_S05_amp03_frameXXXX.png → label = 3 (Right Elbow)`

## Dataset Access

To ensure that the dataset is used **strictly for academic and research purposes**, interested parties are required to complete this request form. Please provide information regarding your **intended use**, **institutional affiliation**, and any **relevant ongoing projects**. Your request will be reviewed, and further instructions will be provided upon approval.

🔗 **Request Form:** [https://forms.gle/z5QGfXP9PxzSZM9F8](https://forms.gle/z5QGfXP9PxzSZM9F8)

---

## Dataset Structure: AJAHR Index & SMPL Mapping

**COCO / MPII Based Annotations**

```
center               → shape: (N, 1, 2),  dtype: float64
scale                → shape: (N, 1),     dtype: float64
imgname              → shape: (N,),       dtype: object
global_orient        → shape: (N, 1, 3, 3), dtype: float32
body_pose            → shape: (N, 72),    dtype: float32
has_body_pose        → shape: (N,),       dtype: float32
betas                → shape: (N, 10),    dtype: float32
has_betas            → shape: (N,),       dtype: float32
body_keypoints_2d    → shape: (N, 25, 3), dtype: float64  # indices 0~24 = body
extra_keypoints_2d   → shape: (N, 19, 3), dtype: float64  # indices 25~43 = extra
body_keypoints_3d    → shape: (N, 25, 4), dtype: float64  # indices 0~24 = body
extra_keypoints_3d   → shape: (N, 19, 4), dtype: float64  # indices 25~43 = extra
body_opt_3d_joints   → shape: (N, 25, 1), dtype: float64
extra_opt_3d_joints  → shape: (N, 19, 1), dtype: float64
```

**H36M Based Annotations**

```
imgname     → shape: (N,),             dtype: <U44>
scale       → shape: (N, 1),           dtype: float64
center      → shape: (N, 1, 2),        dtype: float64
ajahr_conf  → shape: (N, 12),          dtype: int64
global_orient → shape: (N, 1, 1, 3, 3), dtype: float64
gt_2d_kpts  → shape: (N, 1, 44, 3),    dtype: float64  # 0~24 body, 25~43 extra
gt_3d_kpts  → shape: (N, 1, 44, 5),    dtype: float64  # 0~24 body, 25~43 extra
smpl_pose   → shape: (N, 1, 23, 3, 3), dtype: float64
smpl_shape  → shape: (N, 1, 10),       dtype: float64
cam_t       → shape: (N, 1, 3),        dtype: float64
```

---

## 📚 Citation

If you find **AJAHR** useful in your research, please consider citing:

```bibtex
@misc{cho2025ajahramputatedjointaware,
      title={AJAHR: Amputated Joint Aware 3D Human Mesh Recovery},
      author={Hyunjin Cho and Giyun Choi and Jongwon Choi},
      year={2025},
      eprint={2509.19939},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2509.19939},
}
