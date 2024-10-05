import torch
import torch.nn as nn
from HHInter.utils.utils import MotionNormalizerTorch
import numpy as np
import trimesh
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale
        self.normalizer = MotionNormalizerTorch()

    def forward(self, x, timesteps, cond=None, mask=None, motion_cond=None, sdf_points=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)
        if motion_cond is not None:
            motion_cond = torch.cat([motion_cond, torch.zeros_like(motion_cond)], dim=0)
        if sdf_points is not None:
            sdf_points = torch.cat([sdf_points, torch.zeros_like(sdf_points)], dim=0)

        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask, motion_cond=motion_cond, sdf_points=sdf_points)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) *out_uncond

        "Print intermediate marker results for visualization"
        # print("time:", timesteps)
        # motion_output_both = cfg_out.reshape(1, 300, 2, -1)
        # markers = self.normalizer.backward(motion_output_both).detach().cpu().numpy()
        #
        # points_marker = []
        # for fId in [40]:
        #     for idd in range(67):
        #         tfs = np.eye(4)
        #         tfs[:3, 3] = markers[0][fId][0].reshape(-1, 3)[idd]
        #         sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
        #         sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
        #         points_marker.append(sm)
        #
        #         tfs = np.eye(4)
        #         tfs[:3, 3] = markers[0][fId][1].reshape(-1, 3)[idd]
        #         sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
        #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        #         points_marker.append(sm)
        #     points = trimesh.util.concatenate(points_marker)
        #
        #     points.export(r"D:\Motion\Sitcom-Crafter\HHInter\intermediate_res\points_marker_{}.ply".format(timesteps.item()))

        return cfg_out
