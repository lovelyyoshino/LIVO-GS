# 3D Gaussian Splatting for Real-Time Radiance Field Rendering (using results of PlanarSLAM)

This repo is cloned from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). We add additional functions to read **Point Clouds**, **Camera Poses**, and **RGB Images** from results of PlanarSLAM. 
**We hope more possibilities can be explored based on the new Point Clouds**.

## Motivation: Why use results of PlanarSLAM

![planar_points.PNG](assets%2Fplanar_points.PNG)

**New Features of this type of input**
<ol>
<li> Points lying on the non-textured regions </li>
<li> Global plane instances that are represented in different colors </li>
<li> Surface normal vector of every point </li>
</ol>




## BibTex
```commandline
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
```commandline
@inproceedings{Li2021PlanarSLAM,
  author = {Li, Yanyan and Yunus, Raza and Brasch, Nikolas and Navab, Nassir and Tombari, Federico},
  title = {RGB-D SLAM with Structural Regularities},
  year = {2021},
  booktitle = {2021 IEEE international conference on Robotics and automation (ICRA)},
 }
```


## Setup 
We follow the same install steps as suggested by the Original [README](https://github.com/graphdeco-inria/gaussian-splatting) document.
