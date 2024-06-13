from dl_gnn.models.kan_models.model import patch_kan_train_function

patch_kan_train_function()

import argparse
from colorama import Fore, Back, Style
from dl_gnn.models.impl.GNNLF import GNNLF
from dl_gnn.models.impl.ThreeDimFrame import GNNLF as ThreeDGNNLF
from dl_gnn.configs.path_configs import OUTPUT_PATH
from dl_gnn.models.impl import Utils
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn.functional import l1_loss, mse_loss
import numpy as np
import os
import time
from tqdm import tqdm
import wandb
from warnings import filterwarnings


from dl_gnn.tests.Dataset import load
from dl_gnn.utils.visnet_utils import get_dummy_LNNP, gnn_lf_batch2visnet_adapter
from loguru import logger
import pandas as pd


filterwarnings("ignore")

energy_loss_weight = 0.01  # the ratio of energy loss
force_loss_weight = 1  # ratio of force loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="demo", help="molecule in the md17 dataset"
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
    "--use_dir2",
    action="store_true",
    default=True,
    help="whether to do ablation study on one kind of coordinate projections",
)
parser.add_argument(
    "--use_dir3",
    default=True,
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
    default=False,
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
    "--train_batch_size", type=int, default=32, help="batch size for training"
)
parser.add_argument(
    "--test_batch_size", type=int, default=32, help="batch size for testing"
)
parser.add_argument(
    "--validation_batch_size", type=int, default=32, help="batch size for validation"
)
parser.add_argument("--epoches", type=int, default=6000, help="number of epoches")
parser.add_argument(
    "--max_early_stop_steps", type=int, default=6000, help="max early stop steps"
)
parser.add_argument(
    "--colfnet_features",
    action="store_true",
    help="whether to use colfnet features",
    default=False,
)
parser.add_argument(
    "--use_drop_out",
    action="store_true",
    help="whether to use use_drop_out layer in " "the combined features projections",
    default=True,
)
# use_visnet_output_modules
parser.add_argument(
    "--use_visnet_output_modules",
    action="store_true",
    help="whether to use visnet output modules",
    default=False,
)

#         self.use_kan_output_modules = use_kan_output_modules
parser.add_argument(
    "--use_kan_output_modules",
    action="store_true",
    help="whether to use kan output modules",
    default=False,
)

parser.add_argument(
    "--use_visnet_message_passing",
    action="store_true",
    help="whether to use visnet message passing",
    default=False,
)

parser.add_argument(
    "--test_score_steps",
    action="store_true",
    help="the number of steps to test the model",
    default=1000,
)



# use_drop_out
# wandb.login(key="8630da65491fe077fe2c9130bfe5914156ce6f43", relogin=True)
args = parser.parse_args()
proj_name = f"{time.strftime('%m-%d-%H')}-{args.dataset}-{args.modname}-VisNet-Message-Passing-GNNLF"

wandb.init(
    project="GNN-Molecular-GNN-LF-md17",
    name=proj_name,
    mode=args.wandb_mode,
)


MAX_LOSS_TOLERANCE = 500


def init_model(y_mean, y_std, global_y_mean, **kwargs):
    if args.threedframe:
        model = ThreeDGNNLF(
            y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs
        )
    else:
        import json

        with open("model_config.json", "w") as f:
            json.dump(kwargs, f)
        model = GNNLF(**kwargs)
    print(f"numel {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # wandb.log({"numel": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model


def init_combined_model(y_mean, y_std, global_y_mean, **kwargs):
    class GNNLF_AND_VISNET(torch.nn.Module):

        def __init__(self, y_mean, y_std, global_y_mean, **kwargs):
            super(GNNLF_AND_VISNET, self).__init__()
            self.gnnlf = GNNLF(
                y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs
            )
            # self.visnet = get_dummy_LNNP()
            # print the number of parameters in each model
            logger.info(
                f"gnnlf parameters: {sum(p.numel() for p in self.gnnlf.parameters() if p.requires_grad)}"
            )
            # logger.info(f"visnet parameters: {sum(p.numel() for p in self.visnet.parameters() if p.requires_grad)}")
            # 直接使用第二个模型训练到了一定的时候就会出现nan， 我感觉可能是optimizer导致的

        def forward(self, batch):
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # visnet_batch = gnn_lf_batch2visnet_adapter(batch, device)
            z, pos, y, dy = batch
            # pos.requires_grad_(True)
            gnn_lf_pred = self.gnnlf(z, pos)
            # visnet_pred, _ = self.visnet(visnet_batch)
            # pred = (gnn_lf_pred + visnet_pred) / 2
            # pred,_ = self.visnet(visnet_batch)
            return gnn_lf_pred

    model = GNNLF_AND_VISNET(
        y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs
    )
    return model


def load_dataset(dataset_name: str):
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
    patience: int = 10,
    warmup: int = 1,  # 这个不起作用
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
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False
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
    _, atomic_number = train_batch_one[0].shape
    if kwargs.get("atomic_number", None) is None:
        kwargs["atomic_number"] = atomic_number
    y_mean = torch.mean(train_batch_one[2]).item()
    y_std = torch.std(train_batch_one[2]).item()
    global_y_mean = meta_data["global_y_mean"]

    # model = init_model(y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs).to(device)
    model = init_combined_model(
        y_mean=y_mean, y_std=y_std, global_y_mean=global_y_mean, **kwargs
    )

    model = model.to(device)

    # model = torch.compile(model)

    wandb.watch(model, log="all")
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    if not is_training:
        # opt = Adam(model.parameters(), lr=learning_rate * initial_learning_rate_ratio if warmup > 0 else learning_rate)
        opt = AdamW(model.parameters(), lr=0.0004, weight_decay=0.0)

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
        # 用于warmup阶段
        min_lr = learning_rate * minimum_learning_rate_ratio
        scd = ReduceLROnPlateau(opt,
                                "min",
                                0.8,
                                patience=patience,
                                min_lr=min_lr,
                                threshold=0.0001)
        # scd = ReduceLROnPlateau(
        #     opt,
        #     "min",
        #     factor=0.8,
        #     patience=patience,
        #     min_lr=1.0e-08,
        #     # threshold=0.0001
        # )

        # 用于正常训练阶段
        early_stop = 0
        tqdm_epochs = tqdm(range(epoches), desc="Epochs", ncols=140, mininterval=1)
        log_dicts_list = []
        for epoch in tqdm_epochs:
            current_learning_rate = opt.param_groups[0]["lr"]
            training_losses = [[], []]
            start_time = time.time()

            tqdm_object = tqdm(
                train_dataloader,
                desc=f"Training (Epoch {epoch + 1}/{epoches})",
                ncols=140,
                mininterval=1,
            )
            for train_batch in tqdm_object:
                train_batch = [_.to(device) for _ in train_batch]
                training_energy_loss, training_force_loss = (
                    Utils.train_grad_combined_model(
                        train_batch,
                        opt,
                        model,
                        mse_loss,
                        energy_loss_weight,
                        force_loss_weight,
                    )
                )
                # 此时加入新的特征后导致其成为NAN
                if np.isnan(training_force_loss):
                    logger.warning(f"found nan in training force loss, epoch {epoch}")
                    # return NAN_PANITY
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
                print("Current LR:", [group["lr"] for group in opt.param_groups])

            time_cost = time.time() - start_time
            training_energy_loss = np.average(training_losses[0])
            training_force_loss = np.average(training_losses[1])
            train_loss = energy_loss_weight * training_energy_loss + training_force_loss
            validation_energy_loss, validation_force_loss = (
                Utils.test_grad_combined_model(validation_dataloader, model, l1_loss)
            )
            val_loss = (
                energy_loss_weight * validation_energy_loss + validation_force_loss
            )
            early_stop += 1
            if MAX_LOSS_TOLERANCE >= val_loss:
                # 在梯度更新之前裁剪梯度


                scd.step(val_loss)
                print(
                    "Post-Adjustment LR:", [group["lr"] for group in opt.param_groups]
                )

            if np.isnan(val_loss):
                # return NAN_PANITY
                logger.warning(f"found nan in validation loss, epoch {epoch} stop")

            if val_loss < best_val_loss:
                print(
                    f"current loss {val_loss} is better than best loss {best_val_loss}"
                )
                early_stop = 0
                best_val_loss = val_loss
                if save_model:
                    torch.save(model.state_dict(), model_save_path)
            if train_loss < best_train_loss:
                best_train_loss = train_loss

            if early_stop > max_early_stop_steps:
                break
            print(
                f"Epoch: {epoch}, Time elapsed: {time_cost} seconds, "
                f"Learning rate: {current_learning_rate:.4e}, "
                f"Training energy loss: {training_energy_loss:.4f}, Training force loss: {training_force_loss:.4f}, "
                f"Validation energy loss: {validation_energy_loss:.4f}, Validation force loss: {validation_force_loss:.4f}"
            )
            log_dicts = {
                "epoch": epoch,
                "time(s)": time_cost,
                "learning rate": current_learning_rate,
                "training energy loss": training_energy_loss,
                "training force loss": training_force_loss,
                "validation energy loss": validation_energy_loss,
                "validation force loss": validation_force_loss,
            }

            wandb.log(log_dicts)
            wandb.log(
                {"best val loss": best_val_loss, "best train loss": best_train_loss}
            )

            if epoch % 10 == 0:
                print("", end="", flush=True)
            # if training_force_loss > 1000:
            #     return min(best_val_loss, NAN_PANITY)
            if (epoch + 1) % args.test_score_steps == 0:
                print("testing model")
                tst_energy_loss, tst_force_loss = Utils.test_grad_combined_model(
                    test_dataloader, model, l1_loss, show_progress_bar=True
                )
                print(
                    f"Test energy loss: {tst_energy_loss:.4f}, Test force loss: {tst_force_loss:.4f}"
                )
                wandb.log(
                    {
                        "test energy loss": tst_energy_loss,
                        "test force loss": tst_force_loss,
                    }
                )
                log_dicts["test energy loss"] = tst_energy_loss
                log_dicts["test force loss"] = tst_force_loss
            else:
                log_dicts["test energy loss"] = np.nan
                log_dicts["test force loss"] = np.nan
            log_dicts_list.append(log_dicts)
            log_df = pd.DataFrame(log_dicts_list)
            log_df.to_csv(f"{proj_name}.csv")
            # append this to a log file

    if enable_testing:
        model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        mod = model.to(device)
        tst_score = Utils.test_grad_combined_model(
            test_dataloader, mod, l1_loss, show_progress_bar=True
        )
        trn_score = Utils.test_grad_combined_model(train_dataloader, mod, l1_loss)
        val_score = Utils.test_grad_combined_model(validation_dataloader, mod, l1_loss)
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
        OUTPUT_PATH,
        f"{args.dataset}.dirschnet.{args.modname}-{time.strftime('%m-%d-%H')}" f".pt",
    )
    from dl_gnn.tests.md17_params import get_md17_params

    params = get_md17_params(args.dataset)
    params["use_dir2"] = args.use_dir2
    params["use_dir3"] = args.use_dir3
    params["global_frame"] = args.global_frame
    params["no_filter_decomp"] = args.no_filter_decomp
    params["nolin1"] = args.nolin1
    params["no_share_filter"] = args.no_share_filter
    params["colfnet_features"] = args.colfnet_features
    params["use_visnet_output_modules"] = args.use_visnet_output_modules
    params["use_kan_output_modules"] = args.use_kan_output_modules
    params["use_visnet_message_passing"] = args.use_visnet_message_passing
    params["use_drop_out"] = args.use_drop_out
    params["device"] = device
    if args.cutoff is not None:
        params["cutoff"] = args.cutoff
    print(params)
    # wandb.log(
    #     {
    #         "dataset": args.dataset,
    #         "global_frame": args.global_frame,
    #         "no_filter_decomp": args.no_filter_decomp,
    #         "nolin1": args.nolin1,le

    #         "no_share_filter": args.no_share_filter,
    #         "cutoff": args.cutoff
    #     })
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