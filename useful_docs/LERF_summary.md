# LERF Methods

- Problems:
  - **CLIP** creates for global image embedding (over entire input image) and therefore pixel-wise feature extraction becomes difficult
  - querying a single 3D point is ambiguous, its meaning depends on the context and **scale**
- Solution:
  - instead of mapping **CLIP** embeddings to points, learn **CLIP** embeddings for 3D volumes

### NeRF

To get the information (**color**, **density** and **language embedding**) for a single pixel on the screen:

1. from a camera at origin position $o$, shoot out a ray $\vec{r}(t) = \vec{o} + t\vec{d}$, where $t$ is the distance from the camera to the sample along the ray and $\vec d$ is the direction.
   - the MLP storing the **NeRF** returns for each sample point (different values of t) the **color** and the **density** (0 being air, 1 being solid)
2. the resulting list of $n$ **colors** and **densities** for all sample points along the ray are combined into a weighted sum 
   - **Transmittance**: amount of light transmitted to sample $i$ 
      - implemented as weighted sum of **densities** and the **distance** between two consecutive samples $\delta_{j} = t_{j+1} - t_{j}$
        $$T_i = \exp \left( - \sum_{j=1}^{i-1} \sigma_j \delta_j \right) $$
   - multiply **Transmittance** with **light contribution per sample** and **color** and sum over all samples to receive final color
$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \left(1 - \exp(-\sigma_i \delta_i)\right) \mathbf{c}_i $$

### NeRF + Language Field (unsure about scale intuition)
To allow natural language queries, we augment **NeRF**'s output with (view-independent) **language embeddings** $F_{lang} (\vec x,s)$ with input position $\vec x$ and **scale** $s$ (side length in world coordinates):
$$s(t) = s_{img} \times \frac{f_{xy}}{t}$$

- $t$ as position of sample along the ray, $s_{img}$ as initial **scale** of **image plane** (2D-screen where the image is rendered), $f_{xy}$ as **focal length** (Zoom) translating physical $(X, Y, Z)$ into a pixel coordinates $(u, v)$
- results in **frustum** (a cone) with further objects getting a larger **scale**
  - intuition: if you have an image, an object close to the lens with size 10x10 pixels will be tiny (e.g. label on a bottle), but if it's further away (still with size 10x10) it is significantly larger in reality (e.g. car)
  - see **scale** as: $s_{img}$ says 1 cm == 10 pixels, for e.g. $t = 2$ and $f_{xy} = 1$, $s(t)$ will become 1 cm == 5 pixels
    - therefore taking both objects from the previous intuition, the closer one is 1 cm, while the further one is 2 cm in reality, despite both being 10x10 pixels in size

To aggregate all extracted **language embeddings**, the same weights used for **NeRF** are used for a **weighted sum** of all language vectors $\hat{\phi}_{lang} = \int w(t) F_{lang}(\vec{r}(t), s(t)) dt$

Finally this vector is normalized to unit length

### Mutli-Scale Supervision

For efficiency, we precompute an **image pyramid** for every input image over multiple **image crop scales** and store the **CLIP** language embedding of each image crop (as ground truth). The pyramid has $n$ layers with **scales** between $s_{min},s_{max}$ and 50% of each crop overlapping in the grid.

During training a ray is shot and a random scale $s_{img}$ is selected. The returned 3D language vector is then compared with the precomputed ground truth **CLIP** embedding at that **scale**. Due to overlaps, **trilinear interpolation** (avg. of 4 nearest embeddings) creates the ground truth.

The **loss fucntion** $L_{lang} = -\lambda \phi_{lang} \cdot \phi^{gt}_{lang}$ is used in order to maximize the cosine similarity.

**NOTE:** 
- input are multiple images, each of which are precomputed into their own image pyramid
- during training, the rays must come from a known camera postion (otherwise no ground truth)
    - the ground truth embeddings are from the pyramid based on the input image taken at the exact same location
  

### DINO Regularization
Due to **CLIP** embeddings being coarse, **DINO** is used as a secondary supervision to improve clairty. **DINO** works on a per-pixel basis and assigns a **DINO feature** at each point. Therefore no scale parameter is required.

The **DINO feature** $F_{dino}(\vec{x})$ is added as an additional output field and is rendered the same way as the **language embeddings**, but wihtout normalization and instead with **MSE-Loss**.

### Field Architecture

We split the task into two networks:

- **Standard NeRF** to predict **color** and **density**
- **Language Network** to predict **CLIP** and **DINO** vectors

Rationale behind this decision is the unrelatedness between both tasks, as learning the **language and DINO vectors** shouldn't affect the visual representation of the scene.

Both fields will be represented using an **Instant-NGP** (sparse voxel grid to store radiance field)

### Querying LERF

To query **LERF** with a query, a **relevancy score** for a rendered embedding has to be calculated. Additionally a scale needs to be selected automatically for each prompt.

The **relevancy score** is calculated using **CLIP** embedding of the query $\phi_{query}$, the rendered **language embedding** $\phi_{lang}$ and a set of **canonical phrases** $\phi_{canon}$.
$$ \min_{i} \frac{\exp(\phi_{\text{lang}} \cdot \phi_{\text{quer}})}{\exp(\phi_{\text{lang}} \cdot \phi_{\text{canon}}^{i}) + \exp(\phi_{\text{lang}} \cdot \phi_{\text{quer}})} $$
The **canonical phrases** are general terms like "object", "stuff" and using the softmax we check whether the rendered **language vector** is closer to the query or a more general concept.

To automatically select a **scale**, we generate **relevancy maps** for different **scales** (0-2 meters with 30 increments) and select the scale yielding the highest **relevancy score**. This assumes that relevant parts of the scene for a query are at the same **scale**.

### Evaluation
**LERF** is compared to:
- **LSeg (Language-driven Semantic Segmentation)**, a 2D semantic segmentation model which outputs **pixel-wise embeddings** that match the **CLIP** text embedddings. It relies on finetuning using a specific dataset (e.g. **COCO**) and therefore is constrained to the class labels from it's training set. 
- **OwL-ViT (Open-Vocabulary Object Detection with Vision Transformers)**, a 2D open-vocab detector which predicts a box for a query. It work by attaching a lighweight head for classification and bounding box prediction onto a pre-trained 2D image encoder.

#### Qualitative
Visualization of **relevancy scores** $>0.5$, showing hierarchical understanding at different levels (e.g. Cartoon, Elephant). Additionally it's able to find specific characters and query properties. Compared to **LSeg**, **LERF** doesn't have discrete categories, eliminating the issue with out-of-distribution queries.

#### Existence
5 ground truth scenes are labeled using two sets of labels: 1. **COCO** (in-distribution labels for **LSeg**) and 2. long-tail labels (custom objects in the scene, e.g. waldo). **LSeg** performs similar to **LERF** on **COCO** but fails on long-tail labels. Metrics used are **Precision** and **Recall**.

#### Localization
To check the accuracy of **LERF**, novel views are rendered and labeled with 2D bounding boxes for objects. An object was successfully located by **LERF** if its highest relevancy pixel is inside the box, or for **OwL-ViT** if the center of the predicted box is within the box. **LERF** significantly outperforms **OwL-ViT** in **localization accuracy**.

