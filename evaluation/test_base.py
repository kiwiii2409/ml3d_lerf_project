from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict


class BaseTest(ABC):
    def __init__(self, name: str, ground_truth_path: Optional[Path] = None):
        """
        :param name: Name of the test
        :param ground_truth_root: Path to the specific folder/file for this metric's GT
        """
        self.name = name
        self.ground_truth_path = ground_truth_path if ground_truth_path else None




    @abstractmethod
    def load_ground_truth(self, image_dict: Dict):
        pass

    @abstractmethod
    def evaluate_image(self, image_idx: int,rendered_layers: dict, model, gt_meta: dict):
        """eval for a single image"""
        pass

    @abstractmethod
    def summarize(self, verbose):
        pass

    @abstractmethod
    def dump_config(self):
        pass

    @abstractmethod
    def visualize(self):
        """store visualization of rendered view and e.g. sam mask"""
        pass
