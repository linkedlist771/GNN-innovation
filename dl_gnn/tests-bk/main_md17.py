import argparse
from colorama import Fore, Back, Style
from dl_gnn.models.impl.GNNLF import GNNLF
from dl_gnn.models.impl.ThreeDimFrame import GNNLF as ThreeDGNNLF
from dl_gnn.configs.path_configs import OUTPUT_PATH
from dl_gnn.models.impl import Utils
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.functional import l1_loss, mse_loss
import numpy as np
import os
import time
from tqdm import tqdm
import wandb
from warnings import filterwarnings
from Dataset import load

filterwarnings("ignore")

energy_loss_weight = 0.01  # the ratio of energy loss
force_loss_weight = 1  # ratio of force loss
device = torch.device("cuda")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="aspirin", help="molecule in the md17 dataset"
)
parser.add_argument(
    "--modname", type=str, default="0", help="filename used to save model"
)
parser.add_argument(
    "--gemnet_split",
    action="store_true",
    help="whether to use the split of gemnet train/val = 1000/1000",
)
parser.add_argument(
    "--nodir2",
    action="store_true",
    help="whether to do ablation study on one kind of coordinate projections",
)
parser.add_argument(
    "--nodir3",
    action="store_true",
    help="whether to do ablation study on frame-frame projections",
)
parser.add_argument(
    "--global_frame",
    action="store_true",
    help="whether to use a global frame rather than a local frame",
)
parser.add_argument(
    "--no_filter_decomp",
    action="store_true",
    help="whether to do ablation study on filter decomposition",
)
parser.add_argument("--nolin1", action="store_true", help="a hyperparameter")
parser.add_argument(
    "--no_share_filter",
    action="store_true",
    help="whether to do ablation study on sharing filters",
)
parser.add_argument("--cutoff", type=float, default=None, help="cutoff radius")
parser.add_argument("--repeat", type=int, default=1, help="number of repeated runs")
parser.add_argument(
    "--is_training", action="store_true", help="whether to train the model"
)
parser.add_argument(
    "--threedframe",
    action="store_true",
    help="whether to do ablation study on frame ensembles",
)
parser.add_argument(
    "--wandb_mode",
    default="dryrun",
    choices=["online", "dryrun"],
    help="Mode for wandb initialization (online or dryrun)",
)
parser.add_argument(
    "--train_batch_size", type=int, default=120, help="batch size for training"
)
parser.add_argument(
    "--test_batch_size", type=int, default=120, help="batch size for testing"
)
parser.add_argument(
    "--validation_batch_size", type=int, default=120, help="batch size for validation"
)
parser.add_argument("--epoches", type=int, default=6000, help="number of epoches")
parser.add_argument(
    "--max_early_stop_steps", type=int, default=1000, help="max early stop steps"
)
args = parser.parse_args()

wandb.init(
    project="GNN-Molecular-GNN-LF-md17",
    name=f"{time.strftime('%m-%d-%H')}-{args.dataset}-{args.modname}",
    mode=args.wandb_mode,
)


def init_model(y_mean, y_std, global_y_mean, **kwargs):
    if args.threedframe:
        model = ThreeDGNNLF(
            y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs
        )
    else:
        model = GNNLF(y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs)
    print(f"numel {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    wandb.log({"numel": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model


def load_dataset(dataset_name: str):
    # for args.dataset in ['uracil', 'naphthalene', 'aspirin', 'salicylic',  'malonaldehyde', 'ethanol', 'toluene', 'benzene']:
    dataset = load(dataset_name)
    print(Fore.RED + f"current using dataset: {args.dataset}")
    print(Style.RESET_ALL)  # 重置到默认风格
    args.dataset = args.dataset
    N = dataset[0].z.shape[0]
    global_y_mean = torch.mean(dataset.data.y)
    dataset.data.y = (dataset.data.y - global_y_mean).to(torch.float32)
    tensor_dataset = TensorDataset(
        dataset.data.z.reshape(-1, N),
        dataset.data.pos.reshape(-1, N, 3),
        dataset.data.y.reshape(-1, 1),
        dataset.data.dy.reshape(-1, N, 3),
    )
    meta_data = {
        "global_y_mean": global_y_mean,
    }
    return tensor_dataset, meta_data


def train(
    learning_rate: float = 1e-3,
    initial_learning_rate_ratio: float = 1e-1,
    minimum_learning_rate_ratio: float = 1e-3,
    epoches: int = 3000,
    save_model: bool = False,
    enable_testing: bool = False,
    is_training: bool = False,
    search_hp: bool = False,
    max_early_stop_steps: int = 500,
    patience: int = 90,
    warmup: int = 30,
    **kwargs,
):
    tensor_dataset, meta_data = load_dataset(args.dataset)
    NAN_PANITY = 1e1
    # TODO: 这个写法不对吧， 他训练集只有几百个， 测试集有200000。
    # if search_hp:
    #     train_dataset, validation_dataset, test_dataset = random_split(tensor_dataset,
    #                                                                    [950, 256, len(tensor_dataset)-950-256])
    # elif args.gemnet_split:
    #     train_dataset, validation_dataset, test_dataset = random_split(tensor_dataset,
    #                                                                    [1000, 1000, len(tensor_dataset)-2000])
    # else:
    if True:
        train_dataset, validation_dataset, test_dataset = random_split(
            tensor_dataset, [950, 50, len(tensor_dataset) - 1000]
        )
    # train_dataset, validation_dataset, test_dataset = random_split(tensor_dataset,
    #                                                                [int(0.7*len(tensor_dataset)),
    #                                                                  int(0.2*len(tensor_dataset)),
    #                                                                  len(tensor_dataset)-int(0.7*len(tensor_dataset))-int(0.2*len(tensor_dataset))])

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size, shuffle=False
    )
    # 0.1*len(tensor_dataset)])
    # validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)
    # TODO: change the batch size of train_dataset so that the VRAM is enough
    # train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset)//30, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False
    )
    train_data_size = len(train_dataset)

    # trn_dl = Utils.tensorDataloader(train_dataloader, batch_size, True, device)
    train_batch_one = next(iter(train_dataloader))

    y_mean = torch.mean(train_batch_one[2]).item()
    y_std = torch.std(train_batch_one[2]).item()
    global_y_mean = meta_data["global_y_mean"]
    model = init_model(
        y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs
    ).to(device)
    wandb.watch(model, log="all")
    best_val_loss = float("inf")
    if not is_training:
        opt = Adam(
            model.parameters(),
            lr=(
                learning_rate * initial_learning_rate_ratio
                if warmup > 0
                else learning_rate
            ),
        )
        scd1 = StepLR(
            opt,
            1,
            gamma=(
                (1 / initial_learning_rate_ratio)
                ** (1 / (warmup * (train_data_size // args.train_batch_size)))
                if warmup > 0
                else 1
            ),
        )
        scd = ReduceLROnPlateau(
            opt,
            "min",
            0.8,
            patience=patience,
            min_lr=learning_rate * minimum_learning_rate_ratio,
            threshold=0.0001,
        )
        early_stop = 0
        for epoch in range(epoches):
            current_learning_rate = opt.param_groups[0]["lr"]
            training_losses = [[], []]
            start_time = time.time()
            tqdm_object = tqdm(
                train_dataloader, desc=f"Training (Epoch {epoch + 1}/{epoches})"
            )
            for train_batch in tqdm_object:
                train_batch = [_.to(device) for _ in train_batch]
                training_energy_loss, training_force_loss = Utils.train_grad(
                    train_batch,
                    opt,
                    model,
                    mse_loss,
                    energy_loss_weight,
                    force_loss_weight,
                )
                if np.isnan(training_force_loss):
                    return NAN_PANITY
                training_losses[0].append(training_energy_loss)
                training_losses[1].append(training_force_loss)
                # Calculate the average loss for the current batch or over a window
                average_energy_loss = np.mean(
                    training_losses[0][-10:]
                )  # Example: average over the last 10 batches
                average_force_loss = np.mean(training_losses[1][-10:])
                # Update tqdm progress bar with the current loss values
                tqdm_object.set_postfix(
                    energy_loss=average_energy_loss, force_loss=average_force_loss
                )
                if epoch < warmup:
                    scd1.step()

            time_cost = time.time() - start_time
            training_energy_loss = np.average(training_losses[0])
            training_force_loss = np.average(training_losses[1])
            validation_energy_loss, validation_force_loss = Utils.test_grad(
                validation_dataloader, model, l1_loss
            )
            val_loss = (
                energy_loss_weight * validation_energy_loss + validation_force_loss
            )
            early_stop += 1
            scd.step(val_loss)
            if np.isnan(val_loss):
                return NAN_PANITY
            if val_loss < best_val_loss:
                print(
                    f"current loss {val_loss} is better than best loss {best_val_loss}"
                )
                early_stop = 0
                best_val_loss = val_loss
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
            if early_stop > max_early_stop_steps:
                break
            print(
                f"Epoch: {epoch}, Time elapsed: {time_cost} seconds, "
                f"Learning rate: {current_learning_rate:.4e}, "
                f"Training energy loss: {training_energy_loss:.4f}, Training force loss: {training_force_loss:.4f}, "
                f"Validation energy loss: {validation_energy_loss:.4f}, Validation force loss: {validation_force_loss:.4f}"
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "time(s)": time_cost,
                    "learning rate": current_learning_rate,
                    "training energy loss": training_energy_loss,
                    "training force loss": training_force_loss,
                    "validation energy loss": validation_energy_loss,
                    "validation force loss": validation_force_loss,
                }
            )

            if epoch % 10 == 0:
                print("", end="", flush=True)
            if training_force_loss > 1000:
                return min(best_val_loss, NAN_PANITY)

    if enable_testing:
        model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        mod = model.to(device)
        tst_dl = DataLoader(test_dataset, 512)
        tst_score = []
        num_mol = []
        for batch in tst_dl:
            num_mol.append(batch[0].shape[0])
            batch = tuple(_.to(device) for _ in batch)
            tst_score.append(Utils.test_grad(batch, mod, l1_loss))
        num_mol = np.array(num_mol)
        tst_score = np.array(tst_score)
        tst_score = np.sum(tst_score * (num_mol.reshape(-1, 1) / num_mol.sum()), axis=0)
        trn_score = Utils.test_grad(validation_dataloader, mod, l1_loss)
        val_score = Utils.test_grad(validation_dataloader, mod, l1_loss)
        print(trn_score, val_score, tst_score)
        wandb.log(
            {
                "training score": trn_score,
                "validation score": val_score,
                "test score": tst_score,
            }
        )
    print("best val loss", best_val_loss)
    wandb.log({"best val loss": best_val_loss})


if __name__ == "__main__":
    model_save_path = os.path.join(
        OUTPUT_PATH, f"{args.dataset}.dirschnet.{args.modname}.pt"
    )
    from md17_params import get_md17_params

    params = get_md17_params(args.dataset)
    params["use_dir2"] = not args.nodir2
    params["use_dir3"] = not args.nodir3
    params["global_frame"] = args.global_frame
    params["no_filter_decomp"] = args.no_filter_decomp
    params["nolin1"] = args.nolin1
    params["no_share_filter"] = args.no_share_filter
    if args.cutoff is not None:
        params["cutoff"] = args.cutoff
    print(params)
    wandb.log(
        {
            "dataset": args.dataset,
            "use_dir2": not args.nodir2,
            "use_dir3": not args.nodir3,
            "global_frame": args.global_frame,
            "no_filter_decomp": args.no_filter_decomp,
            "nolin1": args.nolin1,
            "no_share_filter": args.no_share_filter,
            "cutoff": args.cutoff,
        }
    )

    for i in range(args.repeat):
        Utils.set_seed(i)
        model_save_path = os.path.join(
            OUTPUT_PATH, f"{args.dataset}.dirschnet.{args.modname}.{i}.pt"
        )
        start_time = time.time()
        train(
            **params,
            # epoches=1,
            epoches=args.epoches,
            max_early_stop_steps=args.max_early_stop_steps,
            save_model=True,
            is_training=args.is_training,
            enable_testing=True,
        )
        wandb.log(
            {
                "total epochs": args.epoches,
                "max early stop steps": args.max_early_stop_steps,
                "save model": True,
                "jump train": args.is_training,
                "do test": True,
            }
        )
        end_time = time.time()
        print(f"repeat {i} / {args.repeat} cost {end_time - start_time} seconds")
        wandb.log({"repeat": i, "cost": end_time - start_time})
