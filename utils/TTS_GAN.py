# Adapted from https://github.com/imics-lab/tts-gan
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def weights_init(m, init_type):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif init_type == 'orth':
            nn.init.orthogonal_(m.weight.data)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform(m.weight.data, 1.)
        else:
            raise NotImplementedError('{} unknown inital type'.format(init_type))
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def train_TTS_GAN(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, train_loader, epoch):
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()
    
    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    # for iter_idx, imgs in enumerate(tqdm(train_loader)):
    for iter_idx, imgs in enumerate(train_loader):
        global_steps = 0

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor).cuda(args["gpu"], non_blocking=True)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args["latent_dim"]))).cuda(args["gpu"], non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        
        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_validity = dis_net(fake_imgs)

        # cal loss
        if args["loss"] == 'hinge':
            d_loss = 0
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args["loss"] == 'standard':
            #soft label
            real_label = torch.full((imgs.shape[0],), 0.9, dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0.1, dtype=torch.float, device=real_imgs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss
        elif args["loss"] == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                    d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                    d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                    d_loss += d_real_loss + d_fake_loss
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
        elif args["loss"] == 'wgangp':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args["phi"])
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args["phi"] ** 2)
        elif args["loss"] == 'wgangp-mode':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args["phi"])
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args["phi"] ** 2)
        elif args["loss"] == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args["phi"])
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args["phi"] ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args["loss"])
        d_loss = d_loss/float(args["accumulated_times"])
        d_loss.backward()
        
        if (iter_idx + 1) % args["accumulated_times"] == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
            dis_optimizer.step()
            dis_optimizer.zero_grad()

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % (args["n_critic"] * args["accumulated_times"]) == 0:
            
            for accumulated_idx in range(args["g_accumulated_times"]):
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args["gen_batch_size"], args["latent_dim"])))
                gen_imgs = gen_net(gen_z)
                fake_validity = dis_net(gen_imgs)

                # cal loss
                loss_lz = torch.tensor(0)
                if args["loss"] == "standard":
                    real_label = torch.full((args["gen_batch_size"],), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
                if args["loss"] == "lsgan":
                    if isinstance(fake_validity, list):
                        g_loss = 0
                        for fake_validity_item in fake_validity:
                            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                            g_loss += nn.MSELoss()(fake_validity_item, real_label)
                    else:
                        real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                        # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                        g_loss = nn.MSELoss()(fake_validity, real_label)
                elif args["loss"] == 'wgangp-mode':
                    fake_image1, fake_image2 = gen_imgs[:args["gen_batch_size"]//2], gen_imgs[args["gen_batch_size"]//2:]
                    z_random1, z_random2 = gen_z[:args["gen_batch_size"]//2], gen_z[args["gen_batch_size"]//2:]
                    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
                    torch.abs(z_random2 - z_random1))
                    eps = 1 * 1e-5
                    loss_lz = 1 / (lz + eps)

                    g_loss = -torch.mean(fake_validity) + loss_lz
                else:
                    g_loss = -torch.mean(fake_validity)
                g_loss = g_loss/float(args["g_accumulated_times"])
                g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()
            gen_optimizer.zero_grad()

            # moving average weight
            ema_nimg = args["ema_kimg"] * 1000
            cur_nimg = args["dis_batch_size"] * args["world_size"] * global_steps
            if args["ema_warmup"] != 0:
                ema_nimg = min(ema_nimg, cur_nimg * args["ema_warmup"])
                ema_beta = 0.5 ** (float(args["dis_batch_size"] * args["world_size"]) / max(ema_nimg, 1e-8))
            else:
                ema_beta = args["ema"]
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args["print_freq"] == 0 and args["rank"] == 0:
            # sample_imgs = torch.cat((gen_imgs[:16], real_imgs[:16]), dim=0)

            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args["max_epoch"], iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), ema_beta))
            del gen_imgs
            del real_imgs
            del fake_validity
            del real_validity
            del g_loss
            del d_loss
    return g_loss.item(), d_loss.item()

class TTS_GAN_Generator(nn.Module):
    def __init__(self, seq_len=150, patch_size=15, channels=3, num_classes=9, latent_dim=100, embed_dim=10, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.blocks = Gen_TransformerEncoder(
                         depth=self.depth,
                         emb_size = self.embed_dim,
                         drop_p = self.attn_drop_rate,
                         forward_drop_p=self.forward_drop_rate
                        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

    def forward(self, z):
        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = x + self.pos_embed
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)
        return output
    
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels = 21, patch_size = 16, emb_size = 100, seq_length = 1024):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = patch_size),
            nn.Linear(patch_size*in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((seq_length // patch_size) + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        return x        
        
        
class TTS_GAN_Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=3,
                 patch_size=15,
                 emb_size=50, 
                 seq_length = 150,
                 depth=3, 
                 n_classes=1, 
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
