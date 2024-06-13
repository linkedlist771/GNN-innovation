import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger


from dl_gnn.models.visnet import datasets, models, priors
from dl_gnn.models.visnet.data import DataModule
from dl_gnn.models.visnet.models import output_modules
from dl_gnn.models.visnet.models.utils import act_class_mapping, rbf_class_mapping
from dl_gnn.models.visnet.module import LNNP
from dl_gnn.models.visnet.utils import (
    LoadFromCheckpoint,
    LoadFromFile,
    number,
    save_argparse,
)


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--load-model",
        action=LoadFromCheckpoint,
        help="Restart training using a model checkpoint",
    )  # keep first
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep second

    # training settings
    parser.add_argument("--num-epochs", default=300, type=int, help="number of epochs")
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="How many steps to warm-up over. Defaults to 0 for no warm-up",
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="Patience for lr-schedule. Patience per eval-interval of validation",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.8,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0, help="Weight decay strength"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Stop training after this many epochs without improvement",
    )
    parser.add_argument(
        "--loss-type", type=str, default="MSE", choices=["MSE", "MAE"], help="Loss type"
    )
    parser.add_argument(
        "--loss-scale-y", type=float, default=1.0, help="Scale the loss y of the target"
    )
    parser.add_argument(
        "--loss-scale-dy",
        type=float,
        default=1.0,
        help="Scale the loss dy of the target",
    )
    parser.add_argument(
        "--energy-weight",
        default=1.0,
        type=float,
        help="Weighting factor for energies in the loss function",
    )

    # dataset specific
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        choices=datasets.__all__,
        help="Name of the torch_geometric dataset",
    )
    parser.add_argument(
        "--dataset-arg", default=None, type=str, help="Additional dataset argument"
    )
    parser.add_argument(
        "--dataset-root", default=None, type=str, help="Data storage directory"
    )
    parser.add_argument(
        "--derivative",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, take the derivative of the prediction w.r.t coordinates",
    )
    parser.add_argument(
        "--split-mode", default=None, type=str, help="Split mode for Molecule3D dataset"
    )

    # dataloader specific
    parser.add_argument(
        "--reload", type=int, default=0, help="Reload dataloaders every n epoch"
    )
    parser.add_argument("--batch-size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--inference-batch-size",
        default=None,
        type=int,
        help="Batchsize for validation and tests.",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, multiply prediction by dataset std and add mean",
    )
    parser.add_argument(
        "--splits", default=None, help="Npz with splits idx_train, idx_val, idx_test"
    )
    parser.add_argument(
        "--train-size",
        type=number,
        default=950,
        help="Percentage/number of samples in training set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--val-size",
        type=number,
        default=50,
        help="Percentage/number of samples in validation set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--test-size",
        type=number,
        default=None,
        help="Percentage/number of samples in test set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data prefetch"
    )

    # model architecture specific
    parser.add_argument(
        "--model",
        type=str,
        default="ViSNetBlock",
        choices=models.__all__,
        help="Which model to train",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="Scalar",
        choices=output_modules.__all__,
        help="The type of output model",
    )
    parser.add_argument(
        "--prior-model",
        type=str,
        default=None,
        choices=priors.__all__,
        help="Which prior model to use",
    )
    parser.add_argument(
        "--prior-args",
        type=dict,
        default=None,
        help="Additional arguments for the prior model",
    )

    # architectural specific
    parser.add_argument(
        "--embedding-dimension", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of interaction layers in the model",
    )
    parser.add_argument(
        "--num-rbf",
        type=int,
        default=64,
        help="Number of radial basis functions in model",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=list(act_class_mapping.keys()),
        help="Activation function",
    )
    parser.add_argument(
        "--rbf-type",
        type=str,
        default="expnorm",
        choices=list(rbf_class_mapping.keys()),
        help="Type of distance expansion",
    )
    parser.add_argument(
        "--trainable-rbf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If distance expansion functions should be trainable",
    )
    parser.add_argument(
        "--attn-activation",
        default="silu",
        choices=list(act_class_mapping.keys()),
        help="Attention activation function",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff in model")
    parser.add_argument(
        "--max-z",
        type=int,
        default=100,
        help="Maximum atomic number that fits in the embedding matrix",
    )
    parser.add_argument(
        "--max-num-neighbors",
        type=int,
        default=32,
        help="Maximum number of neighbors to consider in the network",
    )
    parser.add_argument(
        "--reduce-op",
        type=str,
        default="add",
        choices=["add", "mean"],
        help="Reduce operation to apply to atomic predictions",
    )
    parser.add_argument(
        "--lmax", type=int, default=2, help="Max order of spherical harmonics"
    )
    parser.add_argument(
        "--vecnorm-type",
        type=str,
        default="max_min",
        help="Type of vector normalization",
    )
    parser.add_argument(
        "--trainable-vecnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If vector normalization should be trainable",
    )
    parser.add_argument(
        "--vertex-type",
        type=str,
        default="Edge",
        choices=["None", "Edge", "Node"],
        help="If add vertex angle and Where to add vertex angles",
    )

    # other specific
    parser.add_argument(
        "--ngpus",
        type=int,
        default=-1,
        help="Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating point precision",
    )
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Train or inference",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--distributed-backend", default="ddp", help="Distributed backend"
    )
    parser.add_argument(
        "--redirect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Redirect stdout and stderr to log_dir/log",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")',
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=10,
        help="Test interval, one test per n epochs (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save interval, one save per n epochs (default: 10)",
    )

    args = parser.parse_args()

    if args.redirect:
        os.makedirs(args.log_dir, exist_ok=True)
        sys.stdout = open(os.path.join(args.log_dir, "log"), "w")
        sys.stderr = sys.stdout
        logging.getLogger("pytorch_lightning").addHandler(
            logging.StreamHandler(sys.stdout)
        )

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size
    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


args = {
    "load_model": None,
    "conf": "examples/ViSNet-MD17.yml",
    "conf": None,
    "num_epochs": 300,
    "lr_warmup_steps": 0,
    "lr": 1e-4,
    "lr_patience": 10,
    "lr_min": 1e-6,
    "lr_factor": 0.8,
    "weight_decay": 0.0,
    "early_stopping_patience": 30,
    "loss_type": "MSE",
    "loss_scale_y": 1.0,
    "loss_scale_dy": 1.0,
    "energy_weight": 1.0,
    "dataset": datasets.__all__,
    "dataset_arg": "aspirin",
    "dataset_root": "data",
    "derivative": False,
    "split_mode": None,
    "reload": 0,
    "batch_size": 17,
    "inference_batch_size": 17,  # Inferred default from batch_size as specified
    "standardize": False,
    "splits": None,
    "train_size": 950,
    "val_size": 50,
    "test_size": None,
    "num_workers": 4,
    "model": "ViSNetBlock",
    "output_model": "Scalar",
    "prior_model": None,
    "prior_args": None,
    "embedding_dimension": 256,
    "num_layers": 6,
    "num_rbf": 64,
    "activation": "silu",
    "rbf_type": "expnorm",
    "trainable_rbf": False,
    "attn_activation": "silu",
    "num_heads": 8,
    "cutoff": 5.0,
    "max_z": 100,
    "max_num_neighbors": 32,
    "reduce_op": "add",
    "lmax": 2,
    "vecnorm_type": "max_min",
    "trainable_vecnorm": False,
    "vertex_type": "Edge",
    "ngpus": 1,
    "num_nodes": 1,
    "precision": 32,
    "log_dir": "logs",
    "task": "train",
    "seed": 1,
    "distributed_backend": "ddp",
    "redirect": False,
    "accelerator": "gpu",
    "test_interval": 10,
    "save_interval": 10,
}

pl.seed_everything(args["seed"], workers=True)

from torch.utils.data import Subset


def prepare_dataset(self):
    # assert hasattr(self, f"_prepare_{self.hparams['dataset']}_dataset"), f"Dataset {self.hparams['dataset']} not defined"
    dataset_factory = lambda t: getattr(self, f"_prepare_MD17_dataset")()
    self.idx_train, self.idx_val, self.idx_test = dataset_factory(
        self.hparams["dataset"]
    )
    print(self.dataset)
    self.train_dataset = Subset(self.dataset, self.idx_train)
    self.val_dataset = Subset(self.dataset, self.idx_val)
    self.test_dataset = Subset(self.dataset, self.idx_test)

    if self.hparams["standardize"]:
        self._standardize()


#
# DataModule.prepare_dataset = prepare_dataset
# # initialize data module
# data = DataModule(args)
# data._prepare_MD17_dataset()
# data.prepare_dataset()

# mean = None
# std = None
# prior = None
#
# model = LNNP(args, prior_model=prior, mean=mean, std=std)
import torch_geometric


def gnn_lf_batch2visnet_adapter(batch, device):

    z, pos, y, dy = batch
    batch_size = pos.shape[0]
    sep = pos.shape[1]
    visnet_y = y
    visnet_pos = pos.view(-1, pos.size(-1))
    visnet_z = z.view(
        -1,
    )
    visnet_batch = torch.arange(0, batch_size)
    # repeat each of the value in the visnet_batch for sep times
    visnet_batch = visnet_batch.repeat_interleave(sep)
    visnet_dy = dy.view(-1, dy.size(-1))
    ptr = torch.arange(0, batch_size * sep + 1, sep)
    # conver it to DataBatch with key of "y, pos, z, dy, batch, ptr"
    # init a torch_geometric.data.batch.DataBatch
    visnet_y = visnet_y.to(device)
    visnet_pos = visnet_pos.to(device)
    visnet_z = visnet_z.to(device)
    visnet_dy = visnet_dy.to(device)
    visnet_batch = visnet_batch.to(device)
    ptr = ptr.to(device)
    _batch = torch_geometric.data.Batch(
        y=visnet_y,
        pos=visnet_pos,
        z=visnet_z,
        dy=visnet_dy,
        batch=visnet_batch,
        ptr=ptr,
    )

    return _batch
    # from torch
    # torch_geometric.data.batch.DataBatch


def get_dummy_LNNP():
    mean = None
    std = None
    prior = None

    model = LNNP(args, prior_model=prior, mean=mean, std=std)
    return model


if __name__ == "__main__":
    model = get_dummy_LNNP()
    print(model)
