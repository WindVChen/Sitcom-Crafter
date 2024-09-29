import torch

from HHInter.models.utils import *
from HHInter.models.cfg_sampler import ClassifierFreeSampleModel
from HHInter.models.blocks import *
from HHInter.utils.utils import *

from HHInter.models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D  = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        seq_mask = (mask>0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        batch["motion_emb"] = motion_emb

        return batch


class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 sdf_points_res=16,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        # input_feats is one person's feature, so we need to double it
        self.motion_cond_embed = nn.GRU(self.input_feats, self.latent_dim, batch_first=True)

        self.sdf_compress = Conv3DEncode(input_dims=4, output_dims=1)
        self.sdf_embed = MAPEncoder(in_dim=(sdf_points_res//8)**3, h_dim=self.latent_dim, n_blocks=2, actfun="lrelu")

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))


    def forward(self, x, timesteps, mask=None, cond=None, motion_cond=None, sdf_points=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:]

        if mask is not None:
            mask = mask[...,0]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        if motion_cond is not None:
            "Only use the final hidden state of the GRU."
            emb1 = self.motion_cond_embed(motion_cond[..., :self.input_feats])[1].squeeze(0) + emb
            emb2 = self.motion_cond_embed(motion_cond[..., self.input_feats:])[1].squeeze(0) + emb
        else:
            emb1 = emb
            emb2 = emb

        if sdf_points is not None:
            sdf_points = self.sdf_compress(sdf_points)
            emb1 += self.sdf_embed(sdf_points)
            emb2 += self.sdf_embed(sdf_points)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, emb1, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, emb2, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)

        return output



class InterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50", batch_size=1):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        self.sdf_points_res = cfg.SDF_POINTS_RES
        self.batch_size = batch_size

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = InterDenoiser(self.nfeats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                       num_heads=self.num_heads, dropout=self.dropout, activation=self.activation,
                                 cfg_weight=self.cfg_weight, sdf_points_res=self.sdf_points_res)


        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            # Batch size paramter is for SMPLX model initialization in loss calculation part, that is batch_size * seq_len.
            batch_size=self.batch_size,
            # Whether calculate penetration on all smplx vertics or only on markers. The former one will double both the memory and time cost.
            use_vertex_pene=cfg.USE_VERTEX_PENETRATION,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask


    def compute_loss(self, batch):
        cond = batch["cond"]
        motion_cond_length = batch['motion_cond_length'][0]
        x_start = batch["motions"][:, motion_cond_length:]
        marker_cond = batch["motions"][:, :motion_cond_length]
        B,T = batch["motions"].shape[:2]
        sdf_points = batch["sdf_points"]
        feet_height_thresh = batch["feet_height_thresh"]

        if self.cfg.Normalizing:
            marker_cond = MotionNormalizerTorch().forward(marker_cond.reshape(B, 1, 2, -1)).reshape(B, 1, -1)

        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, 0.1)

        # Condition marker frame
        if marker_cond is not None:
            motion_cond, _ = self.mask_cond(marker_cond, 0.1)

        if sdf_points is not None:
            sdf_cond, _ = self.mask_cond(sdf_points, 0.1)

        seq_mask = self.generate_src_mask(x_start.shape[1], batch["motion_lens"]).to(x_start.device)

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            motion_R=batch['motion_R'],
            motion_T=batch['motion_T'],
            motion_feet=batch['motion_feet'],
            sdf_points=sdf_cond,
            feet_height_thresh=feet_height_thresh,
            is_normalized=self.cfg.Normalizing,
            model_kwargs={"mask":seq_mask,
                          "cond":cond,
                          "motion_cond":motion_cond,
                          'sdf_points':sdf_cond,
                          },
        )
        return output

    def forward(self, batch):
        cond = batch["cond"]
        motion_cond_length = batch['motion_cond_length'][0]
        marker_cond = batch["motions"][:, :motion_cond_length]
        sdf_points = batch["sdf_points"]
        B = cond.shape[0]
        T = batch["motion_lens"][0]

        if self.cfg.Normalizing:
            marker_cond = MotionNormalizerTorch().forward(marker_cond.reshape(B, 1, 2, -1)).reshape(B, 1, -1)

        # Condition marker frame
        motion_cond = marker_cond

        sdf_cond = sdf_points

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            # Batch size paramter is for SMPLX model initialization in loss calculation part, that is batch_size * seq_len.
            batch_size=B,
            # Whether calculate penetration on all smplx vertics or only on markers. The former one will double both the memory and time cost.
            use_vertex_pene=self.cfg.USE_VERTEX_PENETRATION,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )

        self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats*2),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":None,
                "cond":cond,
                "motion_cond":motion_cond,
                "sdf_points":sdf_cond,
            },
            x_start=None)
        return {"output":output}
