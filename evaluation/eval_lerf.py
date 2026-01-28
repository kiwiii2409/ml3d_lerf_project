import os
import json
import torch
import argparse
import datetime
from pathlib import Path


from nerfstudio.utils.eval_utils import eval_setup
from test_base import BaseTest
from test_heatmap import Heatmap_Test


class LERF_Evaluation():
    def __init__(self, config_path: Path, prompt_path: Path, output_path: Path):
        self.config_path = config_path
        self.prompt_path = prompt_path
        self.output_path = output_path

        _, self.pipeline, _, _ = eval_setup(
            config_path,
            test_mode="test"
        ) # returns Loaded config, pipeline module, corresponding checkpoint, and step
        self.pipeline.model.eval()

        with open(self.prompt_path, 'r') as f:
            self.prompt_data = json.load(f)

        self.tests = []

    def add_test(self, new_test: BaseTest):
        self.tests.append(new_test)

    def run_tests(self):
        dataset = self.pipeline.datamanager.eval_dataset
        print(self.pipeline.datamanager.train_dataset.image_filenames)
        print(dataset.image_filenames)
        #TODO: add logic to precalculate sam masks for all images using _get_ground_truth fucntion in heatmap test and store there in dict corresponding to image_idx
        # add method as _get_ground_truth and call for each test via loop. either loads or generates GT using separate script
        num_images = len(dataset)
        with torch.no_grad():
            for i in range(num_images):
                print(f"Processing {i + 1}/{num_images} image")
                camera = dataset.cameras[i: i + 1].to(self.pipeline.device)
                image_filename = dataset.image_filenames[i] # returns e.g. /cluster/51/kzhou/datasets/bouquet/images/frame_00177.jpg
                image_idx = str(image_filename)[-9:-4] # just 00177

                key = str(image_idx)
                if key in self.prompt_data:
                    gt_meta = self.prompt_data[key] # in case specific items were specified for 
                else:
                    gt_meta = self.prompt_data.get("default", {})

                if not gt_meta:
                    continue

                # existing items you want to highlight
                positives = gt_meta.get("positives", [])
                # non existing items to make sure the model isn't marking everything
                negatives = gt_meta.get("negatives", [])
                all_prompts = positives + negatives

                if not all_prompts:
                    continue

                self.pipeline.model.image_encoder.set_positives(all_prompts)
                
                ray_bundle = camera.generate_rays(
                    camera_indices=0, keep_shape=True)
                
                # dict with 'rgb' 'depth' 'relevancy_0' 'relevancy_1' ... (one relevancy for each prompt)
                # TODO: add parameter as toggle in get_output_for_camera_ray_bundle to speed up evaluation (as long as we clear memory frequently it should be fine)
                rendered_layers = self.pipeline.model.get_outputs_for_camera_ray_bundle(
                    ray_bundle)
                print(f"Starting test for {i+1} image:")
                for test in self.tests:
                    test.evaluate_image(
                        int(image_idx), self.pipeline.model, gt_meta, rendered_layers)



        self._save_results()

    def _save_results(self):
        test_results = {}
        print()
        print("===== Evaluation finished =====")
        for test in self.tests:
            test_results[test.name] = test.summarize()
            print(f"{test.name} Results:")
            print(test_results[test.name])
            print("-" * 50)
        # TODO add config information for reproducability
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


    # TODO: fix jank. remove first folder from path to avoid ~/project/lerf_code/lerf_code/outputs/bouquet/lerf/2026-01-17_192939/config.yml due to changed working directory
    trimmed_config_path = Path(*config_path.parts[1:])
    trimmed_dataset_path = Path(*dataset_dir.parts[1:])
    print(trimmed_config_path)
    print(trimmed_dataset_path)
    
    evaluator = LERF_Evaluation(trimmed_config_path, prompt_path, output_path)
    evaluator.add_test(Heatmap_Test(
        name="Heatmap-SAM-IoU",
        ground_truth_path=trimmed_dataset_path / "sam",
        relevancy_threshold=0.5
    ))
    print(f"===== Starting Tests: {[test.name for test in evaluator.tests]} =====")
    evaluator.run_tests()


if __name__ == "__main__":
    cli()