import os
import json
import torch
import argparse
import datetime
from pathlib import Path
import cv2
from tqdm import tqdm 
from nerfstudio.utils.eval_utils import eval_setup
from test_base import BaseTest
from test_heatmap import Heatmap_Test


class LERF_Evaluation():
    def __init__(self, config_path: Path, prompt_path: Path, output_path: Path):
        self.config_path = config_path
        self.prompt_path = prompt_path
        self.output_path = output_path

        # returns Loaded config, pipeline module, corresponding checkpoint, and step
        _, self.pipeline, _, _ = eval_setup(config_path,test_mode="test") 
        self.pipeline.model.eval()

        # moved dataset to init, allows earlier groun_truth loading
        self.eval_dataset = self.pipeline.datamanager.eval_dataset 
        
        # loads abs path of images selected for eval {00177: image_data}
        self.eval_images_dict = self._load_images(self.eval_dataset.image_filenames)

        # load pos & neg prompts
        with open(self.prompt_path, 'r') as f:
            self.prompt_data = json.load(f)

        self.tests = []


    # given a list of image paths from dataset.image_filenames load all images into a Dict {00177: image_data}
    def _load_images(self, image_paths):
        loaded_images = {}
        print("Loading evaluation images into memory...")
        for path in image_paths:
            path_str = str(path)
            
            # frame_00177.jpg -> 00177
            stem = path.stem # 'frame_00177'
            if '_' in stem:
                key = stem.split('_')[-1] # returns 00177
            else:
                key = stem
                
            img = cv2.imread(path_str)
            if img is None:
                raise Exception(f"Warning: Could not load {path_str}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            loaded_images[key] = img
            
        print(f"Loaded {len(loaded_images)} images.")
        return loaded_images
    
    def add_test(self, new_test: BaseTest):
        new_test.load_ground_truth(self.eval_images_dict)
        self.tests.append(new_test)
        

    def run_tests(self):
        num_images = len(self.eval_dataset)
        with torch.no_grad():
            for i in tqdm(range(num_images), desc="eval"):
                print(f"Processing {i + 1}/{num_images} image")
                camera = self.eval_dataset.cameras[i: i + 1].to(self.pipeline.device)
                image_filename = self.eval_dataset.image_filenames[i] # returns e.g. /cluster/51/kzhou/datasets/bouquet/images/frame_00177.jpg
                stem = image_filename.stem
                if '_' in stem:
                    image_idx = stem.split('_')[-1]
                else:
                    image_idx = stem

                if image_idx in self.prompt_data:
                    gt_meta = self.prompt_data[image_idx] # in case specific items were specified for 
                else:
                    gt_meta = self.prompt_data.get("default", {})

                # existing items you want to highlight
                positives = gt_meta.get("positives", [])
                # non existing items to make sure the model isn't marking everything
                negatives = gt_meta.get("negatives", [])
                all_prompts = positives + negatives

                if not all_prompts:
                    continue

                self.pipeline.model.image_encoder.set_positives(all_prompts)
                
                ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
                
                # dict with 'rgb' 'depth' 'relevancy_0' 'relevancy_1' ... (one relevancy for each prompt)
                # TODO: add parameter as toggle in get_output_for_camera_ray_bundle to speed up evaluation (as long as we clear memory frequently it should be fine)
                rendered_layers = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
                print(f"Starting test for {i+1} image:")
                for test in self.tests:
                    test.evaluate_image(
                        image_idx, rendered_layers, self.pipeline.model, gt_meta )
                    
        self._save_results()

    def _save_results(self):
        test_results = {}
        print()
        print("===== Evaluation finished =====")
        for test in self.tests:
            test_results[test.name] = test.summarize(verbose=True)
            
            print(f"{test.name} Results:")
            print(test.summarize())
            print("-" * 50)
        meta_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paths": {
                "load_config": str(self.config_path.resolve()),
                "prompt_config": str(self.prompt_path.resolve()),
                "output_path": str(self.output_path.resolve())
            },
            "tests_executed": [test.dump_config() for test in self.tests]
        }

        final_output = {
            "meta": meta_data,
            "results": test_results
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True) # make sure path exists
        with open(self.output_path, 'w') as f:
            json.dump(final_output, f, indent=4)
        print(f"Results saved to {self.output_path}")
        print("=" * 31)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_config", type=Path, required=True,
                        help="Path to config.yml (e.g. lerf_code/outputs/bouquet/lerf/2026-01-17_192939/config.yml)")
    parser.add_argument("--prompt-config", type=Path, default=None)
    parser.add_argument("--output-path", type=Path,
                        default=None)
    args = parser.parse_args()

    # lerf_code/outputs/bouquet/lerf/2026-01-17_192939/config.yml (back to front with parents)
    config_path = args.load_config
    model_timestamp_dir = config_path.parent # 2026-01-17_192939
    dataset_dir = config_path.parents[2] # bouquet
    version_dir = config_path.parents[4] # lerf_code

    dataset_name = dataset_dir.name
    version_name = version_dir.name
    model_timestamp = model_timestamp_dir.name

    curr_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

    # define prompts to test lerf on
    # positives: existing items you want to highlight, negatives: non-existing items to test the model
    if args.prompt_config is None:
        prompt_path = Path(f"../evaluation/prompts/{dataset_name}.json")# resolve to get absolute paths
    else:
        prompt_path = args.prompt_config


    if args.output_path == None:
        output_path = Path(f"../evaluation/results/{curr_timestamp}_{version_name}_{dataset_name}_{model_timestamp}.json")
    else:
        output_path = args.output_path

    # Swtich working directory to the one specified in the --load_config path. Sets the correct working dir for the relative paths in the datamanagers (otherwise files are stored in wrong directory + no cache)
    print(f"===== Setting up Tests =====")
    os.chdir(version_dir)
    print(f"Switching working directory to: {os.getcwd()}")


    # remove first folder from path to avoid ~/project/lerf_code/lerf_code/outputs/bouquet/lerf/2026-01-17_192939/config.yml due to changed working directory
    trimmed_config_path = Path(*config_path.parts[1:])
    trimmed_dataset_path = Path(*dataset_dir.parts[1:])
    
    evaluator = LERF_Evaluation(trimmed_config_path, prompt_path, output_path)
    evaluator.add_test(Heatmap_Test(
        name="Heatmap-SAM-IoU",
        # ground_truth_path=trimmed_dataset_path / "sam", #TODO maybe remove, not reallu used since into of load_grount_truth in Tests
        relevancy_threshold=0.5
    ))
    print(f"===== Starting Tests: {[test.name for test in evaluator.tests]} =====")
    evaluator.run_tests()


if __name__ == "__main__":
    cli()