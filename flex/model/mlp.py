from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Union

import torch
import numpy as np
import sys


def positional_encoding(positions, freqs):
    """
    Return positional_encoding results with frequency freqs.
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def freq_positional_encoding(
    barf_c2f, positions, freqs, progress 
    ): 
    """Apply the coarse-to-fine positional encoding strategy of BARF. 

    Args:
        opt (edict): settings
        input (torch.Tensor): shaps is (B, ..., C) where C is channel dimension
        embedder_fn (function): positional encoding function
        L (int): Number of frequency basis
    returns:
        positional encoding
    """
    
    input_enc = positional_encoding(positions, freqs)

    # [B,...,2NL]
    # coarse-to-fine: smoothly mask positional encoding for BARF
    if barf_c2f is not None and not (barf_c2f[-1]==0.0):
        # set weights for different frequency bands
        start,end = barf_c2f
        alpha = (progress-start)/(end-start)*freqs
        k = torch.arange(freqs,dtype=torch.float32,device=positions.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        # apply weights
        shape = input_enc.shape
        input_enc = (input_enc.view(-1,freqs)*weight).view(*shape)

    return input_enc


class General_MLP(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): t_pe, feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe.

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
        self,
        inChanel: int,
        outChanel: int,
        t_pe: int = 6,
        fea_pe: int = 6,
        pos_pe: int = 6,
        view_pe: int = 6,
        featureC: int = 128,
        n_layers: int = 3,
        use_sigmoid: bool = True,
        zero_init: bool = True,
        use_tanh: bool = False,
        barf_c2f = None,
    ):
        super().__init__()

        self.in_mlpC = inChanel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid
        self.use_tanh = use_tanh
        self.barf_c2f = barf_c2f
        if self.barf_c2f is not None or not (self.barf_c2f[-1]==0.0):
            self.progress = 0.0
        else:
            self.progress = 1.0

        # Whether use these features as inputs
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        if self.use_fea:
            self.in_mlpC += 2 * fea_pe * inChanel
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3

        assert n_layers >= 2  # Assert at least two layers of MLP
        layers = [torch.nn.Linear(self.in_mlpC, featureC), torch.nn.ReLU(inplace=True)]

        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(featureC, outChanel)]
        self.mlp = torch.nn.Sequential(*layers)

        if zero_init:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)


    def get_PE_weights(self):

        if self.barf_c2f is not None and not (self.barf_c2f[-1]==0.0):
            start,end = self.barf_c2f
            alpha = (self.progress-start)/(end-start)*self.fea_pe
            k = torch.arange(self.fea_pe,dtype=torch.float32)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        else:
            weight = torch.ones(self.fea_pe)

        return weight


    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        features: torch.Tensor,
        frame_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward.
        """
        # Collect input data
        indata = [features]
        if self.use_t:
            indata += [frame_time]
            if self.t_pe > 0:
                indata += [freq_positional_encoding(self.barf_c2f, frame_time, self.t_pe, self.progress)]
        if self.use_fea:
            if self.fea_pe > 0:
                indata += [freq_positional_encoding(self.barf_c2f, features, self.fea_pe, self.progress)]
        if self.use_pos:
            indata += [pts]
            if self.pos_pe > 0:
                indata += [freq_positional_encoding(self.barf_c2f, pts, self.pos_pe, self.progress)]
        if self.use_view:
            indata += [viewdirs]
            if self.view_pe > 0:
                indata += [freq_positional_encoding(self.barf_c2f, viewdirs, self.view_pe, self.progress)]
        mlp_in = torch.cat(indata, dim=-1)

        rgb = self.mlp(mlp_in)
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        if self.use_tanh:
            rgb = torch.tanh(rgb)
            

        return rgb



class MLPRender_Fea_late_view(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): t_pe, feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe. View direction PE will only be added at the last layer!

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
        self,
        inChanel: int,
        outChanel: int,
        t_pe: int = 6,
        fea_pe: int = 6,
        pos_pe: int = 6,
        view_pe: int = 6,
        featureC: int = 128,
        n_layers: int = 3,
        use_sigmoid: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()

        self.in_mlpC = inChanel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid

        self.in_mlpC = 2 * self.fea_pe * inChanel + inChanel
        self.in_view = 2 * self.view_pe * 3 + 3
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC + self.in_view, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True)
        )
        self.mlp_view = torch.nn.Sequential(layer3)
        torch.nn.init.constant_(self.mlp_view[-1].bias, 0)

    def forward(
            self,
            pts: torch.Tensor,
            viewdirs: torch.Tensor,
            features: torch.Tensor,
            frame_time: torch.Tensor,
        ) -> torch.Tensor:
        indata = [features]
        if self.fea_pe > 0:
            indata += [
                torch.zeros(
                    [features.shape[0], self.in_mlpC - features.shape[-1]], 
                    device=features.device)
            ]
        indata_view = [viewdirs]
        if self.view_pe > 0:
            indata_view += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        inter_features = self.mlp(mlp_in)
        mlp_view_in = torch.cat([inter_features] + indata_view, dim=-1)
        rgb = self.mlp_view(mlp_view_in)
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        return rgb
