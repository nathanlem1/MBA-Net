"""
Self-attention mechanism is used to draw global dependencies of input feature maps using different relation functions or
operations. Self-attention, also known as intra-attention, is an attention mechanism relating different positions of a
single sequence (or input feature map) in order to compute a representation of the same sequence (or input feature map).

This code implements spatial and channel attention Modules: SAM & CAM

"""

import torch
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange

import pdb


def compute_reindexing_tensor(l, L, device):
    """
        Re-index the relative positional embedding matrix R from using relative shifts to absolute shifts.
    """
    x = torch.arange(l, device=device)[:, None, None]
    i = torch.arange(l, device=device)[None, :, None]
    r = torch.arange(-(L - 1), L, device=device)[None, None, :]
    mask = ((i - x) == r) & ((i - x).abs() <= L)

    return mask.float()


class SAM_module(nn.Module):
    """ SAM module """
    def __init__(self, in_dim, rel_pos_length, relative_pos=True):
        super(SAM_module, self).__init__()
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
            self.rel_rows = nn.Parameter(torch.randn(num_rel_shifts, dim_key))  # Row relative positional embedding (Rh)
            self.rel_columns = nn.Parameter(torch.randn(num_rel_shifts, dim_key))  # Column relative positional
            # embedding (Rw)

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

            Iw = compute_reindexing_tensor(width, L, device)  # For width-only (column-only)
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
            # Iw = compute_reindexing_tensor(width, L, device)  # For width-only (column-only)
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


class CAM_module(nn.Module):
    """ CAM module """
    def __init__(self, in_dim):
        super(CAM_module, self).__init__()

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
        """
        inputs :
            x : input feature maps( B X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X C X C

        Noted that we do not employ convolution layers to embed features before computing relationships of two channels,
        since it can maintain relationship between different channel maps.
        """
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


# Test
if __name__ == '__main__':
    input = torch.FloatTensor(10, 256, 56, 56)
    print('input:', input.shape)

    # SAM
    L = max(input.shape[2], input.shape[3])
    sam_att = SAM_module(input.shape[1], L)
    output = sam_att(input)
    print('sam_attention: ', output.shape)

    # CAM
    cam_att = CAM_module(input.shape[2]*input.shape[3])
    output = cam_att(input)
    print('cam_attention: ', output.shape)

    print('ok')
