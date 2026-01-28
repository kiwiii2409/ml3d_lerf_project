## LERF Evaluation Usage

This script evaluates a trained LERF model against specific text prompts. It automatically generates Ground Truth segmentation masks using SAM (Segment Anything Model) and calculates the Intersection over Union (IoU) of the LERF relevancy maps.

This script evaluates a trained LERF model using self-defined tests. `eval_lerf.py` works as an orchestrator (loads pipeline, images and calls all tests) and the individual `test_...` classes load or generate their own ground truths and run an per-image evaluation. 
Currently, the function to load all the **ground_truths** is called whenever a test is registered in `eval_lerf.py`'s `cli()` fucntion

## 1. Running Evaluation

Run the evaluation script from your **project root** directory. It assumes a folder structure like:

```
project/
├── evaluation/
├── lerf_code/
└── lerf_mod1/
```


### Basic Command

```bash
python evaluation/eval_lerf.py --load_config lerf_code/outputs/bouquet/lerf/2026-01-17_192939/config.yml
```

### Optional Arguments

* `--prompt-config`: Path to a specific JSON file containing prompts. Defaults to `evaluation/prompts/{dataset_name}.json`.
  * the json defines **positives** (objects which exist in the scene) and **negatives** (non-existent objects) 
    * potentially useful for checking existance of items
  * the json defines a default case (same positive and negative promtps for all images) but also allows the user to define specific prompts for each image
* `--output-path`: Custom path for the results JSON. Defaults to `evaluation/results/file_name.json`.

**Example with options:**

```bash
python evaluation/eval_lerf.py \
  --load_config lerf_mod1/outputs/figurines/lerf/2026-01-20_100000/config.yml \
  --prompt-config evaluation/prompts/figurines_custom.json \
  --output-path evaluation/results/my_test_run.json
```

---

## 2. The Heatmap Test (SAM-IoU)

The default test (`Heatmap_Test`) measures how well LERF's heatmap aligns with the actual object.

1. **Ground Truth Generation**: When the test starts, it runs **SAM** on all evaluation images to generate all possible object masks.
2. **Evaluation**: For a given prompt (e.g., "flower"), it thresholds the LERF relevancy map (default 0.5).
3. **Scoring**: It compares the LERF prediction against **all** SAM masks for that image. The score is the **Max IoU** (the best matching mask found).
   - this avoids the scenario of a prompt for a single object in a multi object scene always resulting in bad scores, as the union over the object mask is always larger than the objects heatmap

**Note:** Ensure your SAM weights path & parameters are correctly set in `evaluation/test_heatmap.py`.

---

## 3. How to Add New Tests

To add a custom metric:

1. **Create a new file** (e.g., `evaluation/test_custom.py`).
2. **Inherit from `BaseTest`** and implement the required methods:
3. **Register the test** in `evaluation/eval_lerf.py` inside the `cli()` function:

```python
from evaluation.test_custom import MyCustomTest

# ... inside cli() ...
evaluator.add_test(MyCustomTest(name="My-Metric", params=...))

```
