- password: 
floating-Backward-19-springs


- fix for errors to run ns-install-cli: typing in `lerf.py`, `lerf_datamanager.py` into more precise types
  - `lerf.py:`
  ```py
      hashgrid_layers: Tuple[int, ...] = (12, 12)
      hashgrid_resolutions: Tuple[Tuple[int,int], ...] = ((16, 128), (128, 512))
      hashgrid_sizes: Tuple[int,...] = (19, 19)
  ```
  - `lerf_datamanager.py:`
  ```py
      patch_tile_size_range: Tuple[float, float] = (0.05, 0.5)
  ```
  
- add lines in lerf.py line 196 for loop to avoid error:
  ```py
              if f"relevancy_{i}" not in outputs:
                      continue
  ```

---
### Data processing
- copy dataset.zip to datasets folder: `scp bouquet.zip kzhou@ml3d.vc.in.tum.de:/cluster/51/kzhou/datasets/`
- convert polycam data to proper dataset: `ns-process-data polycam --data path/to/your/downloaded_data.zip --output-dir path/to/output_folder`
- convert raw vidoe to proper dataset (runs colmap to get camera location): `ns-process-data video --data path/to/my/scan.mp4 --output-dir path/to/my/processed_scan --num-frames-target 200`
  - example data uses around 200-500 images

---
### Training 
- start training on bouquet datset with viewer active: `ns-train lerf --data /cluster/51/kzhou/datasets/bouquet --vis viewer`
  - `--experiment-name desk_baseline` to save/ load from `outputs/desk_baseline`
  - forward port **7007** to local machine with **VC51** assigned by salloc: `ssh -L 7007:TUINI15-VC51.vc.in.tum.de:7007 kzhou@ml3d.vc.in.tum.de`

- view finished training result: `ns-viewer --load-config outputs/bouquet/lerf/TIMESTAMP/config.yml`
  - forward port just as before
  - don't forget to switch from rbg to relevancy_0 to see heatmap, and composited_0 to see overlay

- **NOTE** when running different version, e.g. `lerf_mod1`: run `pip install -e .` in the respective code folder beofre to set it as source for `lerf`. otherwise it will run the other versions code

---
### Others:

- rendering an mp4: define keyframe sequence in `Render` tab and export command. Camera path can be reused (stored in `/cluster/.../datasets/dataset_name/camera_paths`)
  - many modifications in lerf.py to move data to cpu whenever possible

- to open a second terminal in the GPU runtime (run on login node): `srun --jobid=9248 --pty bash`
  - get jobid through `squeue`

- see gpu usage (refresh every 1 seconds):
  - `watch -n 1 nvidia-smi`

---


### Performance:
- Baseline: 50k rays per sec, around 40 min training 
- Mod1: 35k rays per sec


---

### Runs (just for me):
- Baseline Bouquet: 
  - Config File: outputs/bouquet/lerf/2026-01-17_174952/config.yml
  - Checkpoint Directory: outputs/bouquet/lerf/2026-01-17_174952/nerfstudio_models 

- Mod1 Bouquet:
  - Config File: outputs/bouquet/lerf/2026-01-17_192939/config.yml          
  - Checkpoint Directory: outputs/bouquet/lerf/2026-01-17_192939/nerfstudio_models



- render path: 
  - `ns-render camera-path --load-config outputs/bouquet/lerf/2026-01-17_192939/config.yml --camera-path-filename /cluster/51/kzhou/datasets/bouquet/camera_paths/2026-01-17-23-36-22.json --output-path renders/bouquet/2026-01-17-23-36-22.mp4`

  - don't forget to set the prompt in the config
    - `nano outputs/bouquet/lerf/TIMESTAMP/config.yml`
