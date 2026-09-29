"""Tests for modules/city_maps.py on a small synthetic dataset, run with python -m tests.test_city_maps"""
import argparse
import shutil
import tempfile
import unittest
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import safetensors.numpy as stnp
import torch
from torch import nn
from torchmetrics import Accuracy, JaccardIndex

from modules.city_maps import DEVICE, NUM_CLASSES, load_model, predict_city, confusion, split_table, class_table
from modules.model import get_network

# Fixed folder instead of a TemporaryDirectory, the dataset keeps the LMDB open and Windows cannot delete it
WORK = Path(tempfile.gettempdir()) / "mlcz_test_city_maps"
HEIGHT, WIDTH, CHANNELS, STRIDE = 128, 160, 12, 32


class OracleNet(nn.Module):
    """Predicts the label stored in channel 3"""

    def forward(self, x):
        return nn.functional.one_hot(x[:, 3].long(), NUM_CLASSES).permute(0, 3, 1, 2).float()


class VotingNet(nn.Module):
    """Windows at column 0 and 64 give {2: 0.6, 8: 0.4}, the ones at 32 and 96 give {8: 0.9, 1: 0.1}"""

    def forward(self, x):
        first = x[:, 4, 0, 0] % 64 == 0
        probs = torch.zeros(len(x), NUM_CLASSES, device=x.device)
        probs[first, 2], probs[first, 8] = 0.6, 0.4
        probs[~first, 8], probs[~first, 1] = 0.9, 0.1
        return probs.log()[:, :, None, None].expand(-1, -1, 64, 64)


def write_dataset():
    """Cut 64x64 windows from random rasters like convert_data.py, leaving out the top left window"""
    rng = np.random.default_rng(0)
    # Values from 1 to 2, so counting the empty corner of the canvas would shift the stretch
    raster = rng.random((CHANNELS, HEIGHT, WIDTH), dtype=np.float32) + 1
    label = rng.integers(0, NUM_CLASSES, (HEIGHT, WIDTH))
    # Channel 3 holds the label for the oracle net and channel 4 the column for the voting net
    raster[3] = label
    raster[4] = np.arange(WIDTH)
    rows = []
    with lmdb.open(str(WORK / "test.lmdb"), map_size=2 ** 24) as env, env.begin(write=True) as txn:
        for r in range(0, HEIGHT - 64 + 1, STRIDE):
            for c in range(0, WIDTH - 64 + 1, STRIDE):
                if r == c == 0:
                    continue
                key = f"Pavia_{len(rows):06d}_{r}_{c}"
                patch = {"data": raster[:, r:r + 64, c:c + 64].copy(), "label": label[r:r + 64, c:c + 64].copy()}
                txn.put(key.encode(), stnp.save(patch))
                rows.append({"patch_id": key, "city": "Pavia", "split": ["train", "validation", "test"][len(rows) % 3],
                             "row_offset": r, "col_offset": c})
    pd.DataFrame(rows).to_parquet(WORK / "test.parquet")
    return label


def save_checkpoint(args, state_dict):
    path = WORK / f"{args.arch_name}_{args.num_channels}.ckpt"
    torch.save({"state_dict": state_dict, "hyper_parameters": {"args": args}}, path)
    return path


class CityMapsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(WORK, ignore_errors=True)
        WORK.mkdir(parents=True)
        cls.label = write_dataset()

    def predict(self, model):
        return predict_city(model, WORK / "test.lmdb", WORK / "test.parquet", "Pavia", batch_size=4)

    def test_oracle(self):
        result = self.predict(OracleNet())
        covered = np.ones((HEIGHT, WIDTH), dtype=bool)
        covered[:32, :32] = False  # Only the left out window reaches this corner
        self.assertEqual((result["rgb"][covered].min(), result["rgb"][covered].max()), (0, 255))
        np.testing.assert_array_equal(result["label"][covered], self.label[covered])
        np.testing.assert_array_equal(result["pred"], result["label"])
        for key in ["rgb", "label", "pred"]:
            self.assertFalse(result[key][~covered].any(), key)
        # Every patch is counted in its own split
        counts = pd.read_parquet(WORK / "test.parquet")["split"].value_counts()
        self.assertEqual({s: cm.sum() for s, cm in result["confusion"].items()}, dict(counts * 64 * 64))
        np.testing.assert_array_equal(split_table(result["confusion"])[["Accuracy", "Mean IoU"]], 1)

    def test_metrics_match_torchmetrics(self):
        # Random labels and predictions with label 0, classes that are only predicted and classes that never appear
        rng = np.random.default_rng(1)
        labels, preds = rng.integers(0, 6, 5000), rng.integers(0, 9, 5000)
        cm = confusion(labels, preds)
        target, pred = torch.from_numpy(labels), torch.from_numpy(preds)
        metric = dict(task="multiclass", num_classes=NUM_CLASSES, ignore_index=0)
        accuracy = Accuracy(average="micro", **metric)(pred, target).item()
        mean_iou = JaccardIndex(average="macro", **metric)(pred, target).item()
        scores = split_table({"test": cm}).loc["test"]
        self.assertAlmostEqual(scores["Accuracy"], accuracy, places=6)
        self.assertAlmostEqual(scores["Mean IoU"], mean_iou, places=6)
        # Classes 1 to 8 appear in the labels or predictions, so they are the rows of the class table
        ious = JaccardIndex(average="none", **metric)(pred, target).numpy()
        np.testing.assert_allclose(class_table(cm)["IoU"], ious[1:9], atol=1e-6)

    def test_probabilities_are_summed(self):
        # Where windows voting 2 and 8 overlap the summed probability picks 8, averaging the class ids would give 5
        pred = self.predict(VotingNet())["pred"]
        self.assertTrue((pred[:, 32:] == 8).all())
        # Left of column 32 only the windows voting 2 reach
        self.assertTrue((pred[32:, :32] == 2).all())

    def test_load_model(self):
        args = argparse.Namespace(arch_name="CustomCNN", num_channels=CHANNELS, num_classes=NUM_CLASSES, dropout=False)
        net = get_network(args.arch_name, CHANNELS, NUM_CLASSES, pretrained=False, dropout=False)
        # Random weights that load_model can only reproduce by really reading them from the checkpoint
        with torch.no_grad():
            for p in net.parameters():
                p.uniform_(-0.1, 0.1)
        state_dict = {f"model.{k}": v for k, v in net.state_dict().items()}
        # Saved by CombinedLoss when training with --class_weights
        state_dict["criterion.ce_loss.weight"] = torch.ones(NUM_CLASSES)

        model, loaded_args = load_model(save_checkpoint(args, state_dict))
        x = torch.rand(2, CHANNELS, 64, 64, device=DEVICE)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(x), net.eval().to(DEVICE)(x)))
        self.assertEqual(vars(loaded_args), vars(args))

        for change in [{"arch_name": "unet"}, {"num_channels": 13}]:
            with self.assertRaises(RuntimeError):
                load_model(save_checkpoint(argparse.Namespace(**{**vars(args), **change}), state_dict))


if __name__ == "__main__":
    unittest.main()
