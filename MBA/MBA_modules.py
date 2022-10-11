"""
Self-attention mechanism is used to draw global dependencies of input feature maps using different relation functions
# or operations. Self-attention, also known as intra-attention, is an attention mechanism relating different positions
# of a single sequence (or input feature map) in order to compute a representation of the same sequence (or input
# feature map).

"""

import torch
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange

import pdb


# ===================
#     MBA Modules
# ===================


# ------ Method 1:- ABD like modules -----------------------------------------------------------------------------------

def compute_reindexing_tensor(l, L, device):
    """
        Re-index the relative positional embedding matrix R from using relative shifts to absolute shifts.
    """
    x = torch.arange(l, device=device)[:, None, None]
    i = torch.arange(l, device=device)[None, :, None]
    r = torch.arange(-(L - 1), L, device=device)[None, None, :]
    mask = ((i - x) == r) & ((i - x).abs() <= L)

    return mask.float()


# PAM module
class PAM1_module(nn.Module):
    def __init__(self, in_dim, rel_pos_length, relative_pos=True):
        super(PAM1_module, self).__init__()
        self.in_channel = in_dim
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(
            torch.zeros(1))  # gamma is initialized as 0 and gradually learns to assign more weight
        self.relative_pos = relative_pos
        self.rel_pos_length = rel_pos_length

        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),  # C
            nn.BatchNorm2d(in_dim // 8),
            nn.ReLU()
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),  # B
            nn.BatchNorm2d(in_dim // 8),
            nn.ReLU()
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),  # D
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

        # Learn relative positional embeddings for incorporating relative positional encodings
        if self.relative_pos:
            num_rel_shifts = 2 * rel_pos_length - 1
            dim_key = in_dim // 8
            self.bnormr = nn.BatchNorm2d(in_dim)  # dim_key
            self.bnormc = nn.BatchNorm2d(in_dim)  # dim_key
            self.rel_rows = nn.Parameter(torch.randn(num_rel_shifts, dim_key))  # Row relative positional embedding
            self.rel_columns = nn.Parameter(torch.randn(num_rel_shifts, dim_key)) # Column relative positional embedding

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        device = x.device

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # Use transpose of attention
        attention_mask = out.view(m_batchsize, C, height, width)

        # Learn relative positional embeddings for incorporating relative positional encodings
        if self.relative_pos:

            # ----- Standard way of implementation -------------------------------
            L = max(height, width)  # height but max(height, width) is easier to set only once!
            Ih = compute_reindexing_tensor(height, L, device)   # For height-only (row-only)
            Ih = Ih.view(height*width, -1)
            proj_queryT = proj_query.permute(0, 2, 1)
            Ph = torch.mm(Ih, self.rel_rows)
            Eh = torch.matmul(proj_value, torch.matmul(Ph, proj_queryT))
            Eh = Eh.view(m_batchsize, C, height, width)
            Eh = self.bnormr(Eh)  # Batch normalization is really important!

            Iw = compute_reindexing_tensor(width, L, device) # For width-only (column-only)
            Iw = Iw.view(height*width, -1)
            Pw = torch.mm(Iw, self.rel_columns)
            Ew = torch.matmul(proj_value, torch.matmul(Pw, proj_queryT))
            Ew = Ew.view(m_batchsize, C, height, width)
            Ew = self.bnormc(Ew)  # Batch normalization is really important!
            # Add them element-wise
            rel_pos_out = Eh + Ew

            # # ----- Implementation using Einstein-notation -----------------------
            # L = max(height, width)
            # Ih = compute_reindexing_tensor(height, L, device)  # For height-only (Row-only)
            # q, v = map(lambda t: rearrange(t, 'n c (x y) -> n c x y', x=height, y=width),
            #            (proj_query.permute(0, 2, 1), proj_value))
            # Ph = einsum('xir,rd->xid', Ih, self.rel_rows)
            # Sh = einsum('ndxy,xid->nixy', q, Ph)
            # Eh = einsum('nixy,neiy->nexy', Sh, v)
            # Eh = self.bnormr(Eh)  # Batch normalization is really important!
            #
            # Iw = compute_reindexing_tensor(width, L, device)  # Column (== width)
            # Pw = einsum('yir,rd->yid', Iw, self.rel_columns)
            # Sw = einsum('ndxy,yid->nixy', q, Pw)
            # Ew = einsum('nixy,neiy->nexy', Sw, v)  # Gives the best result
            # Ew = self.bnormc(Ew)    # Batch normalization is really important!
            # # Add them element-wise
            # rel_pos_out = Ew + Eh
            # # -------------------------------------------------------------------------------

            attention_mask = attention_mask + rel_pos_out.contiguous()  # Add output of relative positional embeddings
            # to attention mask

        gamma = self.gamma.to(attention_mask.device)
        out = gamma * attention_mask + x

        return out


# CAM module
class CAM1_module(nn.Module):
    """
    inputs :
        x : input feature maps( B X C X H X W)
    returns :
        out : attention value + input feature
        attention: B X C X C

    Noted that we do not employ convolution layers to embed features before computing relationships of two channels,
    since it can maintain relationship between different channel maps.
"""

    def __init__(self, in_dim):
        super(CAM1_module, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(
            torch.zeros(1))  # gamma is initialized as 0 and gradually learns to assign more weight

        # self.queryc_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),  # C
        #     nn.BatchNorm2d(in_dim // 8),
        #     nn.ReLU()
        # )
        # self.keyc_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1),  # B
        #     nn.BatchNorm2d(in_dim // 8),
        #     nn.ReLU()
        # )
        # self.valuec_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),  # D
        #     nn.BatchNorm2d(in_dim),
        #     nn.ReLU()
        # )

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        # Without learning any embedding function - this gives better results for channel attention branch
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # Transpose
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)  # Increases performance only slightly over using self.softmax(energy)
        # attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        attention_mask = out.view(m_batchsize, C, height, width)

        # # With learning embedding functions
        # xc = x.view(m_batchsize, C, -1).permute(0, 2, 1).unsqueeze(-1)
        # proj_query = self.queryc_conv(xc).squeeze(-1).permute(0, 2, 1)
        # proj_key = self.keyc_conv(xc).squeeze(-1)
        # energy = torch.bmm(proj_query, proj_key)
        # max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        # energy_new = max_energy_0 - energy
        # attention = self.softmax(energy_new)
        # proj_value = self.valuec_conv(xc).squeeze(-1).permute(0, 2, 1)
        # out = torch.bmm(attention, proj_value)
        # attention_mask = out.view(m_batchsize, C, height, width)

        gamma = self.gamma.to(attention_mask.device)
        out = gamma * attention_mask + x

        return out


# ------ Method 2:- RGA like modules -----------------------------------------------------------------------------------

# PAM module
class PAM2_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, cha_ratio=8, spa_ratio=8, down_ratio=8, use_biDir_relation=True,
                 relative_pos=True):
        super(PAM2_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_biDir_relation = use_biDir_relation
        self.relative_pos = relative_pos

        self.inter_channel = in_channel // cha_ratio  # cha_ratio - s1
        self.inter_spatial = in_spatial // spa_ratio  # spa_ratio - s1

        # Embedding functions for original features
        self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        # Embedding functions for relation (affinity or similarity) features
        if self.use_biDir_relation:
            num_in_channel_s = self.in_spatial * 2
        else:
            num_in_channel_s = self.in_spatial
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_in_channel_s, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

        # Networks for learning attention weights
        num_channel_s = 1 + self.inter_spatial
        # num_channel_s = self.inter_spatial   # For using relations only i.e. without the original feature.
        self.W_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,  # down_ratio - s2
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_s // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        # Embedding functions for modeling relations (affinity or similarity)
        self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        # Learn relative positional embeddings for incorporating relative positional encodings
        if self.relative_pos:
            import numpy as np
            rel_pos_length = int(np.sqrt(in_spatial))
            num_rel_shifts = 2 * rel_pos_length - 1
            dim_key = in_channel // 8
            self.bnorm_r = nn.BatchNorm2d(in_channel)  # dim_key
            self.bnorm_c = nn.BatchNorm2d(in_channel)  # dim_key
            self.rel_rows_r = nn.Parameter(torch.randn(num_rel_shifts, dim_key))  # Row relative positional embedding
            self.rel_columns_c = nn.Parameter(torch.randn(num_rel_shifts, dim_key))  # Column relative positional
            # embedding

    def forward(self, x):
        b, c, h, w = x.size()
        device = x.device

        # Spatial attention
        theta_xs = self.theta_spatial(x)
        phi_xs = self.phi_spatial(x)
        theta_xs = theta_xs.view(b, self.inter_channel, -1)
        theta_xs = theta_xs.permute(0, 2, 1)  # Take a transpose
        phi_xs = phi_xs.view(b, self.inter_channel, -1)
        Gs = torch.matmul(theta_xs, phi_xs)  # Rs - Spatial affinity matrix

        if self.use_biDir_relation:
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)  # Rs for r_ji
            Gs_out = Gs.view(b, h * w, h, w)  # Rs for r_ij
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # Rs for ri = [rij, rji]
            Gs_joint = self.gg_spatial(Gs_joint)  # Reduce dimension of global relation using gg_spatial (relation)
            # embedding function.

            g_xs = self.gx_spatial(x)  # Reduce dimension of x
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # Further reduce g_xs to 1 along the channel dimension.
            # ys = Gs_joint   # Use relations r_i only without x_i; just for comparison only
            ys = torch.cat((g_xs, Gs_joint), 1)  # y_i = [x_i, r_i]

        else:  # Use simple dot product for pairwise relation
            Gs = Gs.view(b, h * w, h, w)  # Reshape
            g_xs = self.gx_spatial(x)  # Reduce dimension of x
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # Further reduce g_xs to 1 along the channel dimension.
            Gs = self.gg_spatial(Gs)  # Reduce dimension of Gs
            # ys = Gs  # Gs for using simple dot product i.e. r_ij
            ys = torch.cat((g_xs, Gs), 1)  # ys = [xi, rij] i.e. include the original feature

        W_ys = self.W_spatial(ys)  # Learn spatial attention

        # out = F.sigmoid(W_ys.expand_as(x)) * x
        out = torch.sigmoid(W_ys.expand_as(x)) * x

        # Learn relative positional embeddings for incorporating relative positional encodings
        if self.relative_pos:
            L = max(h, w)
            Ih = compute_reindexing_tensor(h, L, device)  # Row (== height)  Ir
            q, v = map(lambda t: rearrange(t, 'n c (x y) -> n c x y', x=h, y=w), (theta_xs.permute(0, 2, 1),
                                                                                  x.view(b, c, -1)))
            Ph = einsum('xir,rd->xid', Ih, self.rel_rows_r)  # Pr
            Sh = einsum('ndxy,xid->nixy', q, Ph)  # Sr
            Eh = einsum('nixy,neiy->nexy', Sh, v)
            Eh = self.bnorm_r(Eh)

            Iw = compute_reindexing_tensor(w, L, device)  # Column (== width)
            Pw = einsum('yir,rd->yid', Iw, self.rel_columns_c)
            Sw = einsum('ndxy,yid->nixy', q, Pw)
            Ew = einsum('nixy,neiy->nexy', Sw, v)  # Gives the best result
            Ew = self.bnorm_c(Ew)  # Batch normalization is really important!
            # Add them element-wise
            rel_pos_out = Ew + Eh

            out = torch.sigmoid(W_ys.expand_as(x) + rel_pos_out.contiguous()) * x

        return out


# CAM module
class CAM2_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, cha_ratio=8, spa_ratio=8, down_ratio=8, use_biDir_relation=True):
        super(CAM2_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_biDir_relation = use_biDir_relation

        self.inter_channel = in_channel // cha_ratio  # cha_ratio - s1
        self.inter_spatial = in_spatial // spa_ratio  # spa_ratio - s1

        # Embedding functions for original features
        self.gx_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

        # Embedding functions for relation (affinity or similarity) features
        if self.use_biDir_relation:
            num_in_channel_c = self.in_channel * 2
        else:
            num_in_channel_c = self.in_channel
        self.gg_channel = nn.Sequential(
            nn.Conv2d(in_channels=num_in_channel_c, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        # Networks for learning attention weights
        num_channel_c = 1 + self.inter_channel
        # num_channel_c = self.inter_channel   # For using relations only i.e. without the original feature.
        self.W_channel = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_c // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        # Embedding functions for modeling relations (affinity or similarity)
        self.theta_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )
        self.phi_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
        theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)  # Transpose
        phi_xc = self.phi_channel(xc).squeeze(-1)
        Gc = torch.matmul(theta_xc, phi_xc)  # Rc

        if self.use_biDir_relation:
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)  # r_ji
            Gc_out = Gc.unsqueeze(-1)  # r_ij
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            # yc = Gc_joint   # Use relations r_i only without x_i; just for comparison only
            yc = torch.cat((g_xc, Gc_joint), 1)

        else:  # Use simple dot product for pairwise relation
            Gc = Gc.unsqueeze(-1)
            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            Gc = self.gg_channel(Gc)  # Reduce dimension of Gc.
            # yc = Gc  # Gc for using simple dot product i.e. r_ij without the original feature.
            yc = torch.cat((g_xc, Gc), 1)  # yc = [xi, rij] i.e. include the original feature

        W_yc = self.W_channel(yc).transpose(1, 2)  # Learn channel attention

        # out = F.sigmoid(W_yc) * x
        out = torch.sigmoid(W_yc) * x

        return out


# Test
if __name__ == '__main__':
    input = torch.FloatTensor(10, 256, 56, 56)
    print('input:', input.shape)

    # ------- Method 1:- ABD like ----------------------------
    # PAM1
    L = max(input.shape[2], input.shape[3])
    pam1_att = PAM1_module(input.shape[1], L)
    output = pam1_att(input)
    print('pam1_att:', output.shape)

    # CAM1
    cam1_att = CAM1_module(input.shape[2]*input.shape[3])
    output = cam1_att(input)
    print('cam1_att:', output.shape)

    # ------- Method 2:- RGA like ----------------------------
    # PAM
    pam2_att = PAM2_Module(input.shape[1], input.shape[2]*input.shape[3], cha_ratio=8, spa_ratio=8, down_ratio=8,
                           use_biDir_relation=True)
    output = pam2_att(input)
    print('pam2_att:', output.shape)

    # CAM
    cam2_att = CAM2_Module(input.shape[1], input.shape[2]*input.shape[3], cha_ratio=8, spa_ratio=8, down_ratio=8,
                           use_biDir_relation=True)
    output = cam2_att(input)
    print('cam2_att:', output.shape)

    print('ok')
