import math
import random
from functools import partial

import torch
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
from src.utils import default, exists

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        cond_drop_prob = 0.5,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.cond_drop_prob = cond_drop_prob
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # class embedding
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, classes, cond_drop_prob = None):
        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        # print("classes", classes.shape)
        classes_emb = self.classes_emb(classes)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b=batch)
            # print("rearrange keep mask", rearrange(keep_mask, 'b -> b 1').shape)
            # print("classes_emb", classes_emb.shape)
            # print("null_classes_emb", null_classes_emb.shape)
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # print("x.shape start", x.shape)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            # print("x.shape block1 downsampling", x.shape)
            h.append(x)

            x = block2(x, t, c)
            # print("x.shape block2 downsampling", x.shape)
            x = attn(x)
            # print("x.shape block2 attn", x.shape)
            h.append(x)

            x = downsample(x)
            # print("x.shape block2 attn", x.shape)

        x = self.mid_block1(x, t, c)
        # print("x.shape middle", x.shape)
        x = self.mid_attn(x)
        # print("x.shape middle", x.shape)
        x = self.mid_block2(x, t,c)
        # print("x.shape middle", x.shape)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            # print("x.shape cat1", x.shape)
            x = block1(x, t, c)
            # print("x.shape block1", x.shape)
            x = torch.cat((x, h.pop()), dim = 1)
            # print("x.shape cat2", x.shape)
            x = block2(x, t ,c)
            # print("x.shape block2", x.shape)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        # print("x.shape cat", x.shape)
        x = self.final_res_block(x, t ,c)
        # print("x.shape final_res_block", x.shape)
        # print(ok)
        return self.final_conv(x)


class Unet1D_more_shares(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        masking_order = 1,
        cond_drop_prob = 0.5,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.masking_order = masking_order
        self.cond_drop_prob = cond_drop_prob
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # class embedding
        print("dim: ", dim)
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim*(self.masking_order+1), classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            rescaled_phi=0.,
            **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, classes, cond_drop_prob = None):
        batch, device = x.shape[0], x.device
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        # print("classes", classes.shape)
        classes_emb = self.classes_emb(classes)
        # print("classes_emb: ", classes_emb.shape)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device=device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b c d', b=batch, c = self.masking_order+1)
            # print("rearrange keep mask", rearrange(keep_mask, 'b -> b 1 1').shape) #just true and false to see which one you want to have the class
            # print("rearrange keep mask", rearrange(keep_mask, 'b -> b 1 1'))
            # print("classes_emb", classes_emb.shape)
            # print("null_classes_emb", null_classes_emb.shape)
            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1 1'),
                classes_emb,
                null_classes_emb
            )
        # print("classes_emb", classes_emb.shape)
        classes_emb = torch.reshape(classes_emb, (classes_emb.shape[0], -1))
        # print("classes_emb", classes_emb.shape)
        c = self.classes_mlp(classes_emb)
        # print("c", c.shape)
        for block1, block2, attn, downsample in self.downs:
            # print("x.shape downsampling", x.shape)
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t,c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        # print("x.shape middle", x.shape)
        x = self.mid_attn(x)
        # print("x.shape middle", x.shape)
        x = self.mid_block2(x, t,c)
        # print("x.shape middle", x.shape)
        for block1, block2, attn, upsample in self.ups:
            # print("x.shape upsampling", x.shape)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t ,c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t ,c)
        return self.final_conv(x)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class MLP(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(MLP, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]

        self.layers = nn.ModuleList()

        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(num_sample_pts, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x



class CNN(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(CNN, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]
        self.conv_layers = search_space["conv_layers"]

        self.layers = nn.ModuleList()
        #CNN
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(search_space)
        num_features = num_sample_pts
        for layer_index in range(0, self.conv_layers):
            #Convolution layer
            new_out_channels = self.filters[layer_index]
            if layer_index == 0:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features,kernel_size = self.kernels[layer_index], stride = self.kernels[layer_index], padding = self.paddings[layer_index])
                if new_num_features <=0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))

            else:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=self.kernels[layer_index],
                                                       stride=self.kernels[layer_index],
                                                       padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))
            #Activation Function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            #Pooling Layer
            if self.pooling_type[layer_index] == "max_pool":
                layer_pool_size = self.pooling_sizes[layer_index]
                layer_pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_maxpool1d(num_features, layer_pool_size, layer_pool_stride)

                if new_num_features <= 0:
                    layer_pool_size = 1
                    layer_pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.MaxPool1d(kernel_size=layer_pool_size, stride=layer_pool_stride))
            elif self.pooling_type[layer_index] == "average_pool":
                pool_size = self.pooling_sizes[layer_index]
                pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_avgpool1d(num_features, pool_size, pool_stride)
                if new_num_features <= 0:
                    pool_size = 1
                    pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))
            #BatchNorm
            self.layers.append(nn.BatchNorm1d(new_out_channels))
            prev_out_channels = new_out_channels
        #MLP
        self.layers.append(nn.Flatten())
        #Flatten
        flatten_neurons =prev_out_channels*num_features
        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(flatten_neurons, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))
            #Activation layer
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x


def cal_num_features_conv1d(n_sample_points,kernel_size, stride,padding = 0, dilation = 1):
        L_in = n_sample_points
        L_out = math.floor(((L_in +(2*padding) - dilation *(kernel_size -1 )-1)/stride )+1)
        return L_out


def cal_num_features_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_num_features_avgpool1d(n_sample_points,kernel_size, stride, padding = 0):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size ) / stride) + 1)
    return L_out


def create_cnn_hp(search_space):
    pooling_type = search_space["pooling_types"]
    pool_size = search_space["pooling_sizes"] #size == stride
    conv_layers = search_space["conv_layers"]
    init_filters = search_space["filters"]
    init_kernels = search_space["kernels"] #stride = kernel/2
    init_padding = search_space["padding"] #only for conv1d layers.
    kernels = []
    strides = []
    filters = []
    paddings = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []
    for conv_layers_index in range(1, conv_layers + 1):
        if conv_layers_index == 1:
            filters.append(init_filters)
            kernels.append(init_kernels)
            strides.append(int(init_kernels / 2))
            paddings.append(init_padding)
        else:
            filters.append(filters[conv_layers_index - 2] * 2)
            kernels.append(kernels[conv_layers_index - 2] // 2)
            strides.append(int(kernels[conv_layers_index - 2] // 4))
            paddings.append(init_padding)
        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)
    return kernels, strides, filters, pooling_type, pooling_sizes, pooling_strides, paddings




def weight_init(m, type = 'kaiming_uniform_'):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type == 'xavier_uniform_':
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('selu'))
        elif type == 'he_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif type == 'random_uniform':
            nn.init.uniform_(m.weight)
        if m.bias != None:
            nn.init.zeros_(m.bias)




def create_hyperparameter_space(model_type):
    if model_type == "mlp":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                                   "lr": random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                                    "optimizer": random.choice( ["RMSprop", "Adam"]),
                                                    "layers": random.randrange(1, 4, 1),
                                                    "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                                    "activation": random.choice(  ["relu", "selu", "elu", "tanh"]),
                                                    "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                                                }
        return search_space
    elif model_type == "cnn":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                              "lr":random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                              "optimizer":random.choice(["RMSprop", "Adam"]),
                                              "layers": random.randrange(1, 4, 1),
                                              "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                              "activation": random.choice( ["relu", "selu", "elu", "tanh"]),
                                              "kernel_initializer": random.choice( ["random_uniform", "glorot_uniform", "he_uniform"]),
                                              "pooling_types": random.choice(["max_pool", "average_pool"]),
                                              "pooling_sizes":random.choice(  [2,4,6,8,10]), #size == strides
                                              "conv_layers": random.choice( [1,2,3,4]),
                                              "filters": random.choice( [4,8,12,16]),
                                              "kernels": random.choice( [i for i in range(26,53,2)]), #strides = kernel/2
                                              "padding": random.choice(  [0,4,8,12,16]),
                                        }

        return search_space