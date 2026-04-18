## Goal
Attempt to improve implementation of LERF by complementing CLIP supervision using Segment-Anything (SAM) to receive clearer object boundaries and prevent the heatmap from bleeding outside of the object. Evaluation using localization accuracy (high-relevancy pixel within hand-labeled bounding box) and/or IoU between generated relevacny maps and SAM-generated masks from different test-views.

## Method (Implementation in folderlerf_mod1)
- separate background and object using SAM masks to obtain separate foreground and background CLIP embeddings
- during training: pass precise coordinates of sampled ray and return fore-/background embedding 
