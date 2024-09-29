import sys
import time

sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from HHInter.datasets import DataModule
from HHInter.configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from HHInter.models import *
from HHInter.global_path import *

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg, model_cfg, data_cfg, sdf_points_res):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(os.path.dirname(__file__), self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Save cfgs to meta dir
        with open(pjoin(self.meta_dir, 'train_cfg.yaml'), 'w') as f:
            f.write(cfg.dump())
        with open(pjoin(self.meta_dir, 'model_cfg.yaml'), 'w') as f:
            f.write(model_cfg.dump())
        with open(pjoin(self.meta_dir, 'data_cfg.yaml'), 'w') as f:
            f.write(data_cfg.dump())

        self.model = model

        self.model.decoder.diffusion.body_regressor.load_state_dict(torch.load(
            get_smplx_body_regressor_checkpoint_path(),
            map_location=self.device)['model_state_dict'])

        # Set body_regressor to eval() mode
        self.model.decoder.diffusion.body_regressor.eval()

        # Feeze this part.
        for param in self.model.decoder.diffusion.body_regressor.parameters():
            param.requires_grad = False

        if self.model.decoder.diffusion.smplx_model is not None:
            self.model.decoder.diffusion.smplx_model.eval()
            for param in self.model.decoder.diffusion.smplx_model.parameters():
                param.requires_grad = False

        self.last_time = -1
        self.verbose = False
        self.writer = SummaryWriter(self.log_dir)

        "======================"
        # Construct sdf points
        self.sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149.
        self.ceiling_height = 3.
        self.sdf_points_res = sdf_points_res

        x = torch.linspace(-self.sdf_points_extents, self.sdf_points_extents, self.sdf_points_res)
        y = torch.linspace(-self.sdf_points_extents, self.sdf_points_extents, self.sdf_points_res)
        z = torch.linspace(-self.ceiling_height, self.ceiling_height, self.sdf_points_res)

        x, y, z = torch.meshgrid(x, y, z)
        # This will fail to transfer to cuda.
        # self.points_scene_coord = torch.stack([x, y, z], dim=-1).permute(3, 0, 1, 2).to(self.device)
        self.register_buffer("points_scene_coord", torch.stack([x, y, z], dim=-1).permute(3, 0, 1, 2))

    def _configure_optim(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        name, text, motion1, motion2, motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points = batch_data
        motion1 = motion1  # .to(self.device)
        motion2 = motion2  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()
        batch["motion_cond_length"] = motion_cond_length.long()
        batch["motion_R"] = motion_R.type(torch.float32)
        batch["motion_T"] = motion_T.type(torch.float32)
        batch["motion_feet"] = motion_feet.long()
        batch["feet_height_thresh"] = feet_height_thresh.type(torch.float32)
        batch["sdf_points"] = torch.cat([self.points_scene_coord.unsqueeze(0).expand(B, -1, -1, -1, -1), sdf_points.type(torch.float32)], dim=1).detach()

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        if self.verbose:
            torch.cuda.synchronize()
            start = time.time()
            if self.last_time != -1:
                print("Batch preparation Time cost: ", start - self.last_time)
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()

        if self.verbose:
            torch.cuda.synchronize()
            print("cuda memory cost: ", torch.cuda.memory_allocated() / 1024 / 1024, "MB")

        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        if self.verbose:
            torch.cuda.synchronize()
            self.last_time = time.time()
            print("Forward total cost: ", self.last_time - start)

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            # Also log learning rate.
            self.writer.add_scalar("lr", self.optimizers().param_groups[0]['lr'], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])



    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()


    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


def build_models(cfg, batch_size):
    "batch size paramter is for SMPLX model initialization in loss calculation part, that is batch_size * seq_len."
    if cfg.NAME == "Story-HIM":
        model = InterGen(cfg, batch_size)
    return model


if __name__ == '__main__':
    print(os.getcwd())
    model_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/model.yaml"))
    train_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/train.yaml"))
    data_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/datasets.yaml")).interhuman

    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS, model_cfg.SDF_POINTS_RES)
    model = build_models(model_cfg, train_cfg.TRAIN.BATCH_SIZE)


    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        # print not matched weight
        for k in model.state_dict().keys():
            if k not in ckpt["state_dict"]:
                print("Not match: ", k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Checkpoint state from {train_cfg.TRAIN.RESUME} loaded!")
    litmodel = LitTrainModel(model, train_cfg, model_cfg, data_cfg, model_cfg.SDF_POINTS_RES)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices="auto", accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        # if only a single gpu is used, comment the following line
        # strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="gloo" if sys.platform == "win32" else "nccl"),
        precision=32,
        callbacks=[checkpoint_callback],

    )

    trainer.fit(model=litmodel, datamodule=datamodule)
