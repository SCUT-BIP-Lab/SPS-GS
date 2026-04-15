# scene/gaussian_model.py
# Copyright (C) 2023
# See LICENSE.md

import os
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# Optional deps
try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except Exception:
    pyg_nn, Data = None, None
    _PYG_AVAILABLE = False

try:
    import kornia
    _KORNIA_AVAILABLE = True
except Exception:
    _KORNIA_AVAILABLE = False

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, args=None):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # GNN
        self.gnn = None
        self.gnn_type = None
        self.gnn_residual_weight = None
        self.gnn_edge_index = None
        self.gnn_last_build_iter = -1
        self.last_gnn_survival_scores = None

        # Protection / aux
        self._creation_iter = torch.empty(0, dtype=torch.long, device="cuda")
        self._is_protected = torch.empty(0, dtype=torch.bool, device="cuda")
        self.origin_xyz = torch.empty(0)
        self.confidence = torch.empty(0)
        self.live_count = torch.empty(0)

        # Global visibility counters and SH band caching
        self._vis_count = torch.empty(0, dtype=torch.int32, device="cuda")
        self._sh_band_index = None

    def capture(self):
        state = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
        if self.gnn is not None:
            return state + (self.gnn.state_dict(),)
        return state

    def restore(self, model_args, training_args):
        if len(model_args) == 12:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
        else:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             gnn_dict) = model_args
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)
            if self.gnn is not None:
                self.gnn.load_state_dict(gnn_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.origin_xyz = fused_point_cloud.clone()
        self.confidence = torch.ones_like(opacities, device="cuda")
        self.live_count = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.int, device="cuda")
        self._creation_iter = torch.zeros(fused_point_cloud.shape[0], dtype=torch.long, device="cuda")
        self._is_protected = torch.zeros(fused_point_cloud.shape[0], dtype=torch.bool, device="cuda")

        self._vis_count = torch.zeros(fused_point_cloud.shape[0], dtype=torch.int32, device="cuda")

    def _edgeconv_mlp(self, in_channels, out_channels):
        hidden = int(getattr(self.args, "gnn_hidden", 64)) if self.args is not None else 64
        mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
        )
        return mlp.cuda()

    def _build_gnn(self, training_args):
        if not _PYG_AVAILABLE or not getattr(training_args, "use_gnn", False):
            self.gnn = None
            self.gnn_type = None
            return

        self.gnn_type = getattr(training_args, "gnn_type", "gatv2").lower()
        hidden = int(getattr(training_args, "gnn_hidden", 64))
        heads = int(getattr(training_args, "gnn_heads", 4))
        layers = int(getattr(training_args, "gnn_layers", 3))
        dropout = float(getattr(training_args, "gnn_dropout", 0.0))
        use_edge_attr = bool(getattr(training_args, "gnn_use_edge_attr", True))

        node_feature_dim = 3 + 3 + 4 + 1 + 3 * ((self.max_sh_degree + 1) ** 2)
        edge_feature_dim = 3 + 1 + 1
        num_sh_total = (self.max_sh_degree + 1) ** 2
        delta_dim = 1 + 3 * num_sh_total + 1

        layers_list = nn.ModuleList()
        in_dim = node_feature_dim

        if self.gnn_type == "gatv2":
            for _ in range(max(0, layers - 1)):
                layers_list.append(
                    pyg_nn.GATv2Conv(
                        in_dim, hidden, heads=heads,
                        edge_dim=edge_feature_dim if use_edge_attr else None,
                        dropout=dropout
                    )
                )
                in_dim = hidden * heads
            layers_list.append(
                pyg_nn.GATv2Conv(
                    in_dim, delta_dim, heads=1, concat=False,
                    edge_dim=edge_feature_dim if use_edge_attr else None,
                    dropout=dropout
                )
            )

        elif self.gnn_type == "gcn":
            for _ in range(max(0, layers - 1)):
                layers_list.append(pyg_nn.GCNConv(in_dim, hidden))
                in_dim = hidden
            layers_list.append(pyg_nn.GCNConv(in_dim, delta_dim))

        elif self.gnn_type == "edgeconv":
            for _ in range(max(0, layers - 1)):
                layers_list.append(pyg_nn.EdgeConv(self._edgeconv_mlp(in_dim, hidden), aggr="max"))
                in_dim = hidden
            layers_list.append(pyg_nn.EdgeConv(self._edgeconv_mlp(in_dim, delta_dim), aggr="max"))

        else:
            print(f"[GNN] Unknown gnn_type='{self.gnn_type}', disabling GNN.")
            self.gnn = None
            self.gnn_type = None
            return

        self.gnn = layers_list.cuda()
        if self.gnn_residual_weight is None:
            self.gnn_residual_weight = nn.Parameter(torch.tensor(0.1, device="cuda"))

    def _apply_gnn_layer(self, layer, x, data, use_edge_attr=True):
        if self.gnn_type == "gatv2":
            if use_edge_attr and getattr(data, "edge_attr", None) is not None:
                return layer(x, data.edge_index, edge_attr=data.edge_attr)
            else:
                return layer(x, data.edge_index)
        elif self.gnn_type == "gcn":
            return layer(x, data.edge_index)
        elif self.gnn_type == "edgeconv":
            return layer(x, data.edge_index)
        else:
            return layer(x, data.edge_index)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Build GNN according to flags
        self._build_gnn(training_args)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if _PYG_AVAILABLE and getattr(training_args, "use_gnn", False) and self.gnn is not None:
            l.append({'params': list(self.gnn.parameters()), 'lr': training_args.gnn_lr, "name": "gnn"})
            if self.gnn_residual_weight is not None:
                l.append({'params': [self.gnn_residual_weight], 'lr': training_args.gnn_lr / 10, "name": "gnn_residual"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        valid_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
        for group in self.optimizer.param_groups:
            if group["name"] not in valid_names:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if "exp_avg" in stored_state and "exp_avg_sq" in stored_state:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
            if stored_state is not None:
                self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, iter: int = None):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self._creation_iter.numel() == mask.numel():
            self._creation_iter = self._creation_iter[valid_points_mask]
            self._is_protected = self._is_protected[valid_points_mask]
        if self.origin_xyz.numel() == self._xyz.numel():
            self.origin_xyz = self.origin_xyz[valid_points_mask]
        if self.confidence.numel() == self._opacity.numel():
            self.confidence = self.confidence[valid_points_mask]
        if self.live_count.numel() != 0 and self.live_count.shape[0] == mask.shape[0]:
            self.live_count = self.live_count[valid_points_mask]
        if self._vis_count.numel() == mask.shape[0]:
            self._vis_count = self._vis_count[valid_points_mask]

        self.gnn_edge_index = None
        self.gnn_last_build_iter = -1

    def opacity_decay(self, factor=0.99):
        opacity = self.get_opacity * factor
        self._opacity.data = self.inverse_opacity_activation(opacity)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in tensors_dict:
                continue
            extension_tensor = tensors_dict[name]
            assert len(group["params"]) == 1
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None and "exp_avg" in stored_state and "exp_avg_sq" in stored_state:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, creation_iter: int = 0):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.confidence.numel() == 0:
            self.confidence = torch.ones(new_opacities.shape, device="cuda")
        else:
            self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], dim=0)

        if self.origin_xyz.numel() == 0:
            self.origin_xyz = new_xyz.clone()
        else:
            self.origin_xyz = torch.cat((self.origin_xyz, new_xyz.clone()), dim=0)

        num_new = new_xyz.shape[0]
        if self._creation_iter.numel() == 0:
            self._creation_iter = torch.zeros(num_new, dtype=torch.long, device="cuda")
            self._is_protected = torch.ones(num_new, dtype=torch.bool, device="cuda")
        else:
            self._creation_iter = torch.cat((self._creation_iter, torch.full((num_new,), int(creation_iter), dtype=torch.long, device="cuda")), dim=0)
            self._is_protected = torch.cat((self._is_protected, torch.ones(num_new, dtype=torch.bool, device="cuda")), dim=0)

        if self._vis_count.numel() == 0:
            self._vis_count = torch.zeros(num_new, dtype=torch.int32, device="cuda")
        else:
            self._vis_count = torch.cat((self._vis_count, torch.zeros(num_new, dtype=torch.int32, device="cuda")), dim=0)

        self.gnn_edge_index = None
        self.gnn_last_build_iter = -1

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter: int = None, dis_prune: bool = False, split_num: int = 2, grace_period: int = 0):
        if iter is not None and self._creation_iter.numel() > 0:
            elapsed = iter - self._creation_iter
            self._is_protected[elapsed > grace_period] = False

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent, N=split_num)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if self._is_protected.numel() == prune_mask.numel():
            prune_mask = torch.logical_and(prune_mask, ~self._is_protected)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if dis_prune and self.origin_xyz.numel() != 0:
            dis = torch.sqrt(((self.origin_xyz - self.get_xyz.detach()) ** 2).mean(-1))
            dis_prune_mask = (dis > 2).squeeze()
            self.prune_points(dis_prune_mask, iter=iter)
        else:
            self.prune_points(prune_mask, iter=iter)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is None:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    @torch.no_grad()
    def accumulate_visibility(self, visibility_filter: torch.Tensor):
        if self._vis_count.numel() != self._xyz.shape[0]:
            self._vis_count = torch.zeros(self._xyz.shape[0], dtype=torch.int32, device="cuda")
        self._vis_count += visibility_filter.to(self._vis_count.dtype)

    def assemble_gnn_features(self, subsample_indices=None):
        if subsample_indices is None:
            subsample_indices = torch.arange(self._xyz.shape[0], device="cuda")
        xyz = self.get_xyz[subsample_indices].detach()
        scaling = self.get_scaling[subsample_indices].detach()
        rotation = self._rotation[subsample_indices].detach()
        opacity = self.get_opacity[subsample_indices].detach()
        sh_features = self.get_features[subsample_indices].detach().flatten(start_dim=1)
        return torch.cat([xyz, scaling, rotation, opacity, sh_features], dim=1)

    def get_gnn_reconstruction_target(self):
        opacity = self.get_opacity.detach()
        sh_features = self.get_features.detach().flatten(start_dim=1)
        return torch.cat([opacity, sh_features], dim=1)

    def _get_gaussian_normals(self, subsample_indices=None, scaling_override=None):
        def quat_rotate(vectors, quats):
            q_vec = quats[:, 1:]
            q_scalar = quats[:, 0]
            return vectors + 2 * torch.cross(q_vec, torch.cross(q_vec, vectors) + q_scalar.unsqueeze(-1) * vectors)

        if subsample_indices is None:
            scaling = scaling_override if scaling_override is not None else self.get_scaling
            rotation = self.get_rotation
        else:
            scaling = scaling_override if scaling_override is not None else self.get_scaling[subsample_indices]
            rotation = self.get_rotation[subsample_indices]
        quats = F.normalize(rotation, p=2, dim=1)
        _, min_scaling_idx = torch.min(scaling.abs() + 1e-8, dim=1)
        normals = torch.zeros_like(scaling)
        normals[torch.arange(normals.shape[0]), min_scaling_idx] = 1.0
        inv_quats = torch.cat([quats[:, :1], -quats[:, 1:]], dim=1)
        normals = quat_rotate(normals, inv_quats)
        return F.normalize(normals, p=2, dim=1)

    def get_full_covariance(self, scaling_modifier=1.0):
        scaling = self.scaling_activation(self._scaling) * scaling_modifier
        rotation = self.rotation_activation(self._rotation)
        S = torch.zeros((scaling.shape[0], 3, 3), dtype=torch.float, device="cuda")
        S[:, 0, 0] = scaling[:, 0]; S[:, 1, 1] = scaling[:, 1]; S[:, 2, 2] = scaling[:, 2]
        R = build_rotation(rotation)
        Sigma = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
        return Sigma

    def get_mahalanobis_neighbors_optimized(self, k=10, candidate_k=100, euclidean_batch_size=1024, subsample_indices=None):
        if subsample_indices is None:
            subsample_indices = torch.arange(self.get_xyz.shape[0], device="cuda")
        num_points = subsample_indices.shape[0]
        points = self.get_xyz.detach()[subsample_indices]
        if num_points <= k + 1:
            return torch.empty((0, k), dtype=torch.long, device="cuda")

        all_candidate_indices = []
        for i_start in range(0, num_points, euclidean_batch_size):
            i_end = min(i_start + euclidean_batch_size, num_points)
            batch_points_i = points[i_start:i_end]
            dists_sq = torch.cdist(batch_points_i, points, p=2) ** 2
            _, batch_candidate_indices = torch.topk(dists_sq, min(candidate_k, num_points), largest=False, dim=1)
            all_candidate_indices.append(batch_candidate_indices)
            del batch_points_i, dists_sq, batch_candidate_indices
        candidate_indices = torch.cat(all_candidate_indices, dim=0)
        del all_candidate_indices
        torch.cuda.empty_cache()

        covariances = self.get_full_covariance().detach()[subsample_indices]
        reg = 1e-6 * torch.eye(3, device="cuda").unsqueeze(0)
        inv_covariances = torch.inverse(covariances + reg)
        del covariances, reg

        mahalanobis_batch_size = 1024
        all_final_neighbors = []
        for i_start in range(0, num_points, mahalanobis_batch_size):
            i_end = min(i_start + mahalanobis_batch_size, num_points)
            batch_points_i = points[i_start:i_end]
            batch_inv_cov_i = inv_covariances[i_start:i_end]
            batch_candidate_indices = candidate_indices[i_start:i_end]
            candidate_points_j = points[batch_candidate_indices]

            diff = candidate_points_j - batch_points_i.unsqueeze(1)
            temp = torch.matmul(diff.unsqueeze(-2), batch_inv_cov_i.unsqueeze(1))
            dists_sq = torch.matmul(temp, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            del batch_points_i, batch_inv_cov_i, candidate_points_j, diff, temp

            original_indices = torch.arange(i_start, i_end, device="cuda").unsqueeze(1)
            self_mask = (batch_candidate_indices == original_indices)
            dists_sq[self_mask] = float('inf')
            del original_indices, self_mask

            _, top_k_in_candidate = torch.topk(dists_sq, k, largest=False, dim=1)
            final_neighbor_indices = torch.gather(batch_candidate_indices, 1, top_k_in_candidate)
            del dists_sq, top_k_in_candidate

            all_final_neighbors.append(final_neighbor_indices)
            del final_neighbor_indices

        result = torch.cat(all_final_neighbors, dim=0)
        del all_final_neighbors, candidate_indices, inv_covariances, points, subsample_indices
        torch.cuda.empty_cache()
        return result

    def build_gnn_graph(self, subsample_ratio=0.25, k=8):
        if not _PYG_AVAILABLE or self._xyz.shape[0] < max(4, k + 1):
            return None, None, None
        if subsample_ratio < 1.0:
            selection_mask = torch.rand(self._xyz.shape[0], device="cuda") < subsample_ratio
            subsample_indices = torch.where(selection_mask)[0]
        else:
            subsample_indices = torch.arange(self._xyz.shape[0], device="cuda")
        if subsample_indices.numel() < k:
            return None, None, None

        points_sub = self._xyz[subsample_indices]
        neighbor_indices_in_subsample = self.get_mahalanobis_neighbors_optimized(k=k, subsample_indices=subsample_indices)
        if neighbor_indices_in_subsample is None or neighbor_indices_in_subsample.numel() == 0:
            return None, None, None

        src = torch.arange(points_sub.shape[0], device="cuda").unsqueeze(1).expand_as(neighbor_indices_in_subsample).flatten()
        dst = neighbor_indices_in_subsample.flatten()
        edge_index = torch.stack([src, dst], dim=0)

        with torch.no_grad():
            clamped_scaling = self.get_scaling[subsample_indices].detach().clamp(min=1e-6)
            normals_sub = self._get_gaussian_normals(subsample_indices, scaling_override=clamped_scaling)
            rel_pos = points_sub[src] - points_sub[dst]
            dist = torch.norm(rel_pos, dim=1, keepdim=True)
            normal_cons = torch.sum(normals_sub[src] * normals_sub[dst], dim=1, keepdim=True)
            edge_attr = torch.cat([rel_pos, dist, normal_cons], dim=1)
            edge_attr = (edge_attr - edge_attr.mean(dim=0, keepdim=True)) / (edge_attr.std(dim=0, keepdim=True) + 1e-6)

        features_sub = self.assemble_gnn_features(subsample_indices)
        data = Data(x=features_sub, edge_index=edge_index, edge_attr=edge_attr, pos=points_sub)
        return data, subsample_indices, neighbor_indices_in_subsample

    def _line_hole_overlap(self, uv1, uv2, hole_mask, n_samples=64):
        H, W = hole_mask.shape
        u1, v1 = uv1; u2, v2 = uv2
        ts = torch.linspace(0, 1, n_samples, device=hole_mask.device)
        us = (u1 * (1 - ts) + u2 * ts).round().long()
        vs = (v1 * (1 - ts) + v2 * ts).round().long()
        valid = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        if not valid.any():
            return 0.0
        return float(hole_mask[vs[valid], us[valid]].float().mean().item())

    def _project_world_to_cam(self, cam, pts_world):
        if hasattr(cam, "get_focal"):
            fx, fy = cam.get_focal()
        else:
            fx, fy = cam.focal_x, cam.focal_y
        if hasattr(cam, "get_principal"):
            cx, cy = cam.get_principal()
        else:
            cx, cy = cam.image_width * 0.5, cam.image_height * 0.5
        if hasattr(cam, "world_view_transform"):
            w2c = cam.world_view_transform
        else:
            w2c = torch.eye(4, device=pts_world.device, dtype=torch.float32)
        N = pts_world.shape[0]
        homo = torch.cat([pts_world, torch.ones(N, 1, device=pts_world.device)], dim=1)
        cam_h = (w2c @ homo.T).T
        Xc, Yc, Zc = cam_h[:, 0], cam_h[:, 1], cam_h[:, 2].clamp(min=1e-6)
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy
        return torch.stack([u, v], dim=1), Zc

    def _ensure_sh_band_index(self):
        if self._sh_band_index is not None:
            return self._sh_band_index
        D = self.max_sh_degree
        bands = []
        for b in range(1, D + 1):
            bands.extend([b] * (2 * b + 1))
        self._sh_band_index = torch.tensor(bands, dtype=torch.float32, device="cuda")
        return self._sh_band_index

    def refine_with_gnn(self, visibility_filter, iteration, opt):
        if not _PYG_AVAILABLE or self.gnn is None:
            return torch.tensor(0.0, device="cuda")

        data, subsample_indices, _ = self.build_gnn_graph(opt.gnn_subsample_ratio, opt.gnn_k)
        if data is None:
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device="cuda")

        with torch.no_grad():
            m, s = data.x.mean(dim=0, keepdim=True), data.x.std(dim=0, keepdim=True)
            data.x = (data.x - m) / (s + 1e-6)

        if not torch.all(torch.isfinite(data.x)) or not torch.all(torch.isfinite(data.edge_attr)):
            del data
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device="cuda")

        x = data.x
        use_edge_attr = bool(getattr(opt, "gnn_use_edge_attr", True))

        for i, layer in enumerate(self.gnn):
            xi = x
            xo = self._apply_gnn_layer(layer, xi, data, use_edge_attr=use_edge_attr)
            xa = torch.tanh(xo)
            x = xa + xi if xa.shape == xi.shape else xa
            if i < len(self.gnn) - 1:
                x = F.leaky_relu(x, negative_slope=0.2)
            del xi, xo, xa

        gnn_out = x
        del data
        torch.cuda.empty_cache()

        if not torch.all(torch.isfinite(gnn_out)):
            del gnn_out
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device="cuda")

        predicted_appearance, _ = torch.split(gnn_out, [gnn_out.shape[1] - 1, 1], dim=1)
        original_appearance = self.get_gnn_reconstruction_target()[subsample_indices]
        subsample_visibility_mask = visibility_filter[subsample_indices]

        loss_visible = torch.tensor(0.0, device="cuda")
        if subsample_visibility_mask.any():
            loss_visible = F.mse_loss(predicted_appearance[subsample_visibility_mask],
                                      original_appearance[subsample_visibility_mask])
        gnn_losses = loss_visible

        if not getattr(opt, "gnn_apply_updates", False):
            del gnn_out, original_appearance, subsample_indices, subsample_visibility_mask
            torch.cuda.empty_cache()
            return gnn_losses

        predicted_deltas = predicted_appearance - original_appearance
        invisible_mask = ~subsample_visibility_mask
        target_mask = invisible_mask if getattr(opt, "gnn_update_only_invisible", True) else torch.ones_like(invisible_mask)
        if self._vis_count.numel() == self._xyz.shape[0]:
            target_mask = torch.logical_or(target_mask, (self._vis_count[subsample_indices] <= 2))
        vis_ratio = visibility_filter.sum().float() / max(1, self._xyz.shape[0])
        if vis_ratio > float(getattr(opt, "gnn_vis_ratio_thresh", 1.1)):
            target_mask = torch.zeros_like(target_mask)

        with torch.no_grad():
            if target_mask.any():
                _, delta_sh = torch.split(predicted_deltas, [1, predicted_deltas.shape[1] - 1], dim=1)
                num_sh_total = (self.max_sh_degree + 1) ** 2
                delta_rest = delta_sh[:, 3:].view(-1, num_sh_total - 1, 3).clamp(-0.05, 0.05)

                band = self._ensure_sh_band_index().view(1, -1, 1)
                delta_rest = delta_rest * (1.0 / (1.0 + band))

                inv_idx = subsample_indices[target_mask]
                cur_rest = self._features_rest.detach()[inv_idx]
                rms_feat = torch.sqrt(torch.mean(cur_rest ** 2, dim=(1, 2)) + 1e-8)
                rms_delta = torch.sqrt(torch.mean(delta_rest[target_mask] ** 2, dim=(1, 2)) + 1e-8)
                scale = (rms_feat / (rms_delta + 1e-6)).clamp(0.0, 2.0) * 0.25
                scale = scale.view(-1, 1, 1)

                base = (self.gnn_residual_weight if self.gnn_residual_weight is not None
                        else torch.tensor(0.1, device="cuda"))
                blend = base * (1 - vis_ratio)

                self._features_rest.data[inv_idx] += blend * scale * delta_rest[target_mask]

                if getattr(opt, "gnn_update_opacity", False):
                    delta_opacity = (predicted_deltas[:, :1]).clamp(-0.02, 0.02)
                    self._opacity.data[inv_idx] += blend * delta_opacity[target_mask]

        del gnn_out, predicted_deltas, original_appearance, subsample_indices, subsample_visibility_mask
        torch.cuda.empty_cache()
        return gnn_losses

    # === LBO planning (unchanged major logic) ===

    def analyze_geometric_components(self, min_component_size, k=8):
        if not _SCIPY_AVAILABLE:
            return None, None, None
        num_points = self.get_xyz.shape[0]
        if num_points < k:
            return None, None, None
        neighbor_indices = self.get_mahalanobis_neighbors_optimized(k=k)
        if neighbor_indices is None or neighbor_indices.numel() == 0:
            return None, None, None
        source_indices = torch.arange(num_points, device="cuda").unsqueeze(1).expand_as(neighbor_indices)
        u, v = source_indices.flatten(), neighbor_indices.flatten()
        edges_cpu = torch.stack([u[u != v], v[u != v]], dim=1).cpu().numpy()
        if edges_cpu.shape[0] == 0:
            return None, None, None
        adj_matrix = csr_matrix((np.ones(edges_cpu.shape[0]), (edges_cpu[:, 0], edges_cpu[:, 1])), shape=(num_points, num_points))
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        labels_tensor = torch.from_numpy(labels).long().cuda()
        component_sizes = torch.bincount(labels_tensor)
        significant_component_mask = component_sizes >= max(1, int(min_component_size))
        return labels_tensor, component_sizes, significant_component_mask

    def _create_bridging_gaussians(self, points1, points2, count):
        dists = torch.cdist(points1, points2)
        idx = dists.argmin()
        p1_idx = (idx // dists.shape[1]).item()
        p2_idx = (idx % dists.shape[1]).item()
        start_point, end_point = points1[p1_idx], points2[p2_idx]
        alphas = torch.linspace(0, 1, count + 2, device="cuda")[1:-1]
        bridge_points = start_point.unsqueeze(0) * (1 - alphas.unsqueeze(1)) + end_point.unsqueeze(0) * alphas.unsqueeze(1)
        return bridge_points

    def _compute_local_planes(self, neighbor_points):
        if not torch.all(torch.isfinite(neighbor_points)):
            return None, None
        N, k, _ = neighbor_points.shape
        if k < 3:
            return None, None
        centroids = torch.mean(neighbor_points, dim=1)
        centered_points = neighbor_points - centroids.unsqueeze(1)
        cov = torch.bmm(centered_points.transpose(1, 2), centered_points) / (k - 1)
        if not torch.all(torch.isfinite(cov)):
            return None, None
        cov = (cov + cov.transpose(1, 2)) / 2.0
        try:
            _, eigenvectors = torch.linalg.eigh(cov)
        except torch.linalg.LinAlgError:
            return None, None
        normals = eigenvectors[:, :, 0]
        return F.normalize(normals, p=2, dim=-1), centroids

    def _create_filling_gaussians(self, boundary_indices, k):
        if not _SCIPY_AVAILABLE:
            return {}
        from utils.general_utils import matrix_to_quaternion
        if boundary_indices.shape[0] == 0:
            return {}
        neighbor_indices = self.get_mahalanobis_neighbors_optimized(k=k, subsample_indices=boundary_indices)
        if neighbor_indices is None or neighbor_indices.numel() == 0:
            return {}
        neighborhood_points = self._xyz[boundary_indices[neighbor_indices]].detach()
        plane_normals, centroids = self._compute_local_planes(neighborhood_points)
        if centroids is None:
            return {}
        new_xyz = centroids
        neighbor_features = self.get_features[boundary_indices[neighbor_indices]]
        new_features = torch.mean(neighbor_features, dim=1)
        avg_dist = torch.mean(torch.norm(neighborhood_points[:, 0, :].unsqueeze(1) - neighborhood_points[:, 1:, :], dim=2), dim=1)
        new_scales = torch.stack([avg_dist * 0.5, avg_dist * 0.5, avg_dist * 0.1], dim=1)
        target_vec = F.normalize(plane_normals, p=2, dim=1)
        up_vec = torch.tensor([0.0, 1.0, 0.0], device="cuda").expand_as(target_vec).clone()
        colinear = torch.abs(torch.sum(target_vec * up_vec, dim=1)) > 0.99
        up_vec[colinear] = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        right_vec = F.normalize(torch.cross(up_vec, target_vec), p=2, dim=1)
        new_up_vec = F.normalize(torch.cross(target_vec, right_vec), p=2, dim=1)
        rot_mats = torch.stack([right_vec, new_up_vec, target_vec], dim=2)
        new_rotations = matrix_to_quaternion(rot_mats)
        return {'xyz': new_xyz, 'features': new_features, 'scales': new_scales, 'rotations': new_rotations}

    def lbo_gaussian_planning(self, scene, pipe, background, opt, iteration):
        """
        LBO 拓扑规划（与现有逻辑一致），桥接点上限由 lbo_bridge_points_count 控制。
        修复：在选择边界上出现次数最多的两个显著连通分量时的索引错误。
        """
        if not _SCIPY_AVAILABLE:
            return
        from gaussian_renderer import render
        import torch.nn.functional as F

        print(f"\n[ITER {iteration}] Running Unified LBO Gaussian Planning Module...")

        component_labels, component_sizes, sig_mask = self.analyze_geometric_components(
            opt.lbo_planning_min_component_size, k=getattr(opt, "gnn_k", 8)
        )
        if component_labels is None:
            print("[INFO] LBO analysis failed or scene is too small. Skipping planning.")
            return

        prune_mask = ~sig_mask[component_labels]
        num_to_prune = prune_mask.sum().item()

        new_gaussians_planned_data = []

        train_cams = scene.getTrainCameras().copy()
        if not train_cams:
            return

        candidate_cams = [train_cams[0]]
        if len(train_cams) >= 2:
            candidate_cams.append(train_cams[min(1, len(train_cams) - 1)])

        best_cam, max_hole_score = None, -1
        with torch.no_grad():
            for cam in candidate_cams:
                pkg = render(cam, self, pipe, background)
                alpha = pkg.get("rendered_alpha", pkg.get("alpha", None))
                if alpha is None:
                    continue
                if alpha.dim() == 2:
                    a = alpha.unsqueeze(0).unsqueeze(0)
                elif alpha.dim() == 3:
                    a = alpha.unsqueeze(0)
                else:
                    continue
                a_small = F.interpolate(a, size=(64, 64), mode="area").squeeze()
                score = (a_small < 0.5).sum()
                if score > max_hole_score:
                    max_hole_score, best_cam = score, cam

        if best_cam is None:
            return

        with torch.no_grad():
            pkg = render(best_cam, self, pipe, background)
            alpha_map = pkg.get("rendered_alpha", pkg.get("alpha", None))
            depth_map = pkg.get("rendered_depth", pkg.get("depth", None))

        if alpha_map is None or depth_map is None:
            print("[INFO] Renderer does not provide alpha/depth. Skip LBO planning.")
            return

        if alpha_map.dim() == 3:
            alpha_map_hw = alpha_map.squeeze(0)
        elif alpha_map.dim() == 2:
            alpha_map_hw = alpha_map
        else:
            print("[INFO] Unexpected alpha shape, skip.")
            return

        if depth_map.dim() == 3:
            depth_map_hw = depth_map.squeeze(0)
        elif depth_map.dim() == 2:
            depth_map_hw = depth_map
        else:
            print("[INFO] Unexpected depth shape, skip.")
            return

        hole_mask = (alpha_map_hw < 0.5)

        if hole_mask.any() and _KORNIA_AVAILABLE:
            # 边界粗定位（膨胀后与孔洞取差）
            dilated = kornia.morphology.dilation(
                hole_mask.unsqueeze(0).unsqueeze(0).float(),
                torch.ones(5, 5, device=hole_mask.device)
            ).squeeze().bool()
            boundary_pixels_mask = dilated & ~hole_mask

            if boundary_pixels_mask.any():
                boundary_pixels_2d = torch.nonzero(boundary_pixels_mask)
                boundary_depths = depth_map_hw[boundary_pixels_mask]

                H, W = best_cam.image_height, best_cam.image_width

                if hasattr(best_cam, "get_focal"):
                    fx, fy = best_cam.get_focal()
                elif hasattr(best_cam, "focal_x") and hasattr(best_cam, "focal_y"):
                    fx, fy = best_cam.focal_x, best_cam.focal_y
                else:
                    fx = float(W); fy = float(H)

                if hasattr(best_cam, "get_principal"):
                    cx, cy = best_cam.get_principal()
                elif hasattr(best_cam, "cx") and hasattr(best_cam, "cy"):
                    cx, cy = best_cam.cx, best_cam.cy
                else:
                    cx = W * 0.5; cy = H * 0.5

                fx = float(fx); fy = float(fy)
                cx = float(cx); cy = float(cy)

                u = boundary_pixels_2d[:, 1].float()
                v = boundary_pixels_2d[:, 0].float()

                cam_x = (u - cx) * boundary_depths / (fx + 1e-8)
                cam_y = (v - cy) * boundary_depths / (fy + 1e-8)
                cam_z = boundary_depths

                points_cam_space = torch.stack([cam_x, cam_y, cam_z], dim=-1)

                if hasattr(best_cam, "world_view_transform"):
                    c2w = torch.inverse(best_cam.world_view_transform).to(points_cam_space.device, dtype=torch.float32)
                    ones = torch.ones_like(cam_z).unsqueeze(-1)
                    points_homo = torch.cat([points_cam_space, ones], dim=-1)
                    xyz_world_boundary = (c2w @ points_homo.T).T[:, :3]
                else:
                    xyz_world_boundary = points_cam_space

                # 将边界像素对应到最近高斯索引
                kdtree_3d = cKDTree(self._xyz.detach().cpu().numpy())
                _, nn_indices_np = kdtree_3d.query(xyz_world_boundary.detach().cpu().numpy(), k=1)
                boundary_gauss_indices = torch.from_numpy(nn_indices_np).long().to(self._xyz.device)

                # 统计边界上不同连通分量的出现频次
                boundary_component_labels = component_labels[boundary_gauss_indices]
                unique_labels, counts = torch.unique(boundary_component_labels, return_counts=True)

                # 仅保留“显著”的连通分量
                mask_sig_on_boundary = sig_mask[unique_labels]               # bool mask per unique label
                labels_filtered = unique_labels[mask_sig_on_boundary]
                counts_filtered = counts[mask_sig_on_boundary]

                # 若边界上至少出现了两个显著连通分量：桥接它们
                if labels_filtered.numel() >= 2:
                    _, top_idx = torch.topk(counts_filtered, 2)
                    label1 = labels_filtered[top_idx[0]]
                    label2 = labels_filtered[top_idx[1]]

                    pts1_idx = boundary_gauss_indices[boundary_component_labels == label1]
                    pts2_idx = boundary_gauss_indices[boundary_component_labels == label2]
                    pts1 = self._xyz[pts1_idx]
                    pts2 = self._xyz[pts2_idx]

                    if pts1.numel() > 0 and pts2.numel() > 0:
                        dists = torch.cdist(pts1, pts2)
                        flati = torch.argmin(dists)
                        i1 = int(flati // dists.shape[1]); i2 = int(flati % dists.shape[1])

                        cap = getattr(opt, "lbo_bridge_points_count", 30000)
                        bridge_points = self._create_bridging_gaussians(pts1[i1:i1+1], pts2[i2:i2+1], cap)
                        new_gaussians_planned_data.append({'xyz': bridge_points})

                # 若只有一个显著连通分量：执行内部填充
                elif labels_filtered.numel() == 1:
                    label = labels_filtered[0]
                    points_on_boundary = boundary_gauss_indices[boundary_component_labels == label]
                    if points_on_boundary.shape[0] > opt.lbo_planning_max_new_points:
                        perm = torch.randperm(points_on_boundary.shape[0], device=self._xyz.device)
                        points_on_boundary = points_on_boundary[perm[:opt.lbo_planning_max_new_points]]
                    fill_data = self._create_filling_gaussians(points_on_boundary, k=getattr(opt, "gnn_k", 8))
                    if fill_data:
                        new_gaussians_planned_data.append(fill_data)

        if num_to_prune > 0:
            print(f"[LBO Planning] Pruning {num_to_prune} points from insignificant components.")
            self.prune_points(prune_mask, iteration)

        if new_gaussians_planned_data:
            all_new_xyz = torch.cat([d['xyz'] for d in new_gaussians_planned_data if 'xyz' in d], dim=0)
            if all_new_xyz.shape[0] > 0:
                print(f"[LBO Planning] Adding {all_new_xyz.shape[0]} new Gaussians.")
                kdtree = cKDTree(self.get_xyz.detach().cpu().numpy())
                _, nn_indices = kdtree.query(all_new_xyz.detach().cpu().numpy(), k=1)
                nn_indices = torch.from_numpy(nn_indices).long().to(self._xyz.device)

                new_features_dc = self._features_dc[nn_indices]
                new_features_rest = self._features_rest[nn_indices]
                new_opacities = self._opacity[nn_indices] * 0.9
                new_scaling = self._scaling[nn_indices]
                new_rotations = self._rotation[nn_indices]

                self.densification_postfix(
                    all_new_xyz, new_features_dc, new_features_rest,
                    new_opacities, new_scaling, new_rotations, creation_iter=iteration
                )

        torch.cuda.empty_cache()
