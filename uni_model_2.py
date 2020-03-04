import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.autograd import Function
from collections import Counter


class MusicAttrVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 tempo_dims,
                 velocity_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrVAE, self).__init__()
        
        # encoder
        self.gru_r = nn.GRU(roll_dims + 3, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_n = nn.GRU(roll_dims + 3, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_c = nn.GRU(roll_dims + 24, hidden_dims, batch_first=True, bidirectional=True)

        # sub-decoder
        self.gru_d_r = nn.GRU(z_dims + rhythm_dims, hidden_dims, batch_first=True)
        self.gru_d_n = nn.GRU(z_dims + note_dims, hidden_dims, batch_first=True)
        self.gru_d_c = nn.GRU(z_dims + chroma_dims, hidden_dims, batch_first=True)

        # classifiers
        self.c_r = nn.Linear(z_dims, 3)
        self.c_n = nn.Linear(z_dims, 3)
        
        # mu and logvar
        self.mu_r, self.var_r = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_n, self.var_n = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_c, self.var_c = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # global decoder
        num_dims = 3
        cdtl_dims = 3 + 3 + 24
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear init before sub-decoder
        self.linear_init_r = nn.Linear(z_dims, hidden_dims)
        self.linear_init_n = nn.Linear(z_dims, hidden_dims)
        self.linear_init_c = nn.Linear(z_dims, hidden_dims)

        # linear out after sub-decoder
        self.linear_out_r = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_out_n = nn.Linear(hidden_dims, note_dims)
        self.linear_out_c = nn.Linear(z_dims, chroma_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, c_r_oh, c_n_oh, chroma, is_res=False):
        # rhythm encoder
        c_r_oh = torch.stack([c_r_oh] * x.shape[1], dim=1)
        x_r = torch.cat([x, c_r_oh], dim=-1)
        x_r = self.gru_r(x_r)[-1]
        x_r = x_r.transpose_(0, 1).contiguous().view(x_r.size(0), -1)
        mu_r, var_r = self.mu_r(x_r), self.var_r(x_r).exp_()
        
        # note encoder
        c_n_oh = torch.stack([c_n_oh] * x.shape[1], dim=1)
        x_n = torch.cat([x, c_n_oh], dim=-1)
        x_n = self.gru_n(x_n)[-1]
        x_n = x_n.transpose_(0, 1).contiguous().view(x_n.size(0), -1)
        mu_n, var_n = self.mu_n(x_n), self.var_n(x_n).exp_()

        # chroma encoder
        chroma = torch.stack([chroma] * x.shape[1], dim=1)
        x_c = torch.cat([x, chroma], dim=-1)
        x_c = self.gru_c(x_c)[-1]
        x_c = x_c.transpose_(0, 1).contiguous().view(x_c.size(0), -1)
        mu_c, var_c = self.mu_c(x_c), self.var_c(x_c).exp_()

        dis_r = Normal(mu_r, var_r)
        dis_n = Normal(mu_n, var_n)
        dis_c = Normal(mu_c, var_c)

        output = (dis_r, dis_n, dis_c)

        return output

    def sub_decoders(self, rhythm, z_r, note, z_n, z_c):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        rhythm_in = get_hidden_and_concat_latent(rhythm, z_r)
        h_r = self.linear_init_r(z_r).unsqueeze(0)
        rhythm_out = self.gru_d_r(rhythm_in, h_r)[0]
        rhythm_out = F.log_softmax(self.linear_out_r(rhythm_out), 1)

        note_in = get_hidden_and_concat_latent(note, z_n)
        h_n = self.linear_init_n(z_n).unsqueeze(0)
        note_out = self.gru_d_n(note_in, h_n)[0]
        note_out = F.log_softmax(self.linear_out_n(note_out), 1)

        chroma_out = torch.nn.Sigmoid()(self.linear_out_c(z_c))   # BCE

        return rhythm_out, note_out, chroma_out
    
    def global_decoder(self, z, steps):
        out = torch.zeros((z.size(0), self.roll_dims)).cuda()
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = self.linear_init_global(z)
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        
        # if not self.training:
            # print("not training mode")

        for i in range(steps):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_g(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_g_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_g(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                # self.eps = self.k / \
                #     (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma, c_r_oh, c_n_oh,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis_r, dis_n, dis_c = self.encoder(x, c_r_oh, c_n_oh, chroma, is_res=is_res)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        # any difference in result between repar and .rsample()?

        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)
        z_c = repar(dis_c.mean, dis_c.stddev)

        # if is_class, get the classes for each latent
        if is_class:
            cls_r = self.c_r(z_r)
            cls_n = self.c_n(z_n)

        # get sub decoders output
        r_out, n_out, c_out = self.sub_decoders(rhythm, z_r, 
                                                note, z_n, 
                                                z_c)

        # packaging output
        z = torch.cat([z_r, c_r_oh, z_n, c_n_oh, z_c, chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        output = (out, r_out, n_out, c_out)
        dis = (dis_r, dis_n, dis_c)
        
        if is_class:
            clas = (cls_r, cls_n)
        
        res = (output, dis, clas) if is_class else (output, dis)
        return res


class MusicAttrRegVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 tempo_dims,
                 velocity_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrRegVAE, self).__init__()
        
        # encoder
        self.gru_r = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_n = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_c = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)

        # dropouts
        self.e_dropout = nn.Dropout(p=0.3)

        # sub-decoder
        self.gru_d_r = nn.GRU(z_dims + rhythm_dims, hidden_dims, batch_first=True)
        self.gru_d_n = nn.GRU(z_dims + note_dims, hidden_dims, batch_first=True)
        self.gru_d_c = nn.GRU(z_dims + chroma_dims, hidden_dims, batch_first=True)

        # classifiers
        self.c_r = nn.Linear(z_dims, 3)
        self.c_n = nn.Linear(z_dims, 3)
        
        # mu and logvar
        self.mu_r, self.var_r = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_n, self.var_n = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_c, self.var_c = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # global decoder
        num_dims = 2
        cdtl_dims = 24
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear init before sub-decoder
        self.linear_init_r = nn.Linear(z_dims, hidden_dims)
        self.linear_init_n = nn.Linear(z_dims, hidden_dims)
        self.linear_init_c = nn.Linear(z_dims, hidden_dims)

        # linear out after sub-decoder
        self.linear_out_r = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_out_n = nn.Linear(hidden_dims, note_dims)
        self.linear_out_c = nn.Linear(z_dims, chroma_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, c_r_oh, c_n_oh, chroma, is_res=False):
        # rhythm encoder
        x_r = self.gru_r(x)[-1]
        x_r = x_r.transpose_(0, 1).contiguous().view(x_r.size(0), -1)
        # x_r = nn.Dropout(p=0.3)(x_r)
        mu_r, var_r = self.mu_r(x_r), self.var_r(x_r).exp_()
        
        # note encoder
        x_n = self.gru_n(x)[-1]
        x_n = x_n.transpose_(0, 1).contiguous().view(x_n.size(0), -1)
        # x_n = nn.Dropout(p=0.5)(x_n)
        mu_n, var_n = self.mu_n(x_n), self.var_n(x_n).exp_()

        # chroma encoder
        # x_c = self.gru_c(x)[-1]
        # x_c = x_c.transpose_(0, 1).contiguous().view(x_c.size(0), -1)
        # mu_c, var_c = self.mu_c(x_c), self.var_c(x_c).exp_()

        dis_r = Normal(mu_r, var_r)
        dis_n = Normal(mu_n, var_n)
        # dis_c = Normal(mu_c, var_c)

        output = (dis_r, dis_n)

        return output

    def sub_decoders(self, rhythm, z_r, note, z_n):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        rhythm_in = get_hidden_and_concat_latent(rhythm, z_r)
        h_r = self.linear_init_r(z_r).unsqueeze(0)
        rhythm_out = self.gru_d_r(rhythm_in, h_r)[0]
        rhythm_out = F.log_softmax(self.linear_out_r(rhythm_out), 1)

        note_in = get_hidden_and_concat_latent(note, z_n)
        h_n = self.linear_init_n(z_n).unsqueeze(0)
        note_out = self.gru_d_n(note_in, h_n)[0]
        note_out = F.log_softmax(self.linear_out_n(note_out), 1)

        # chroma_out = torch.nn.Sigmoid()(self.linear_out_c(z_c))   # BCE

        return rhythm_out, note_out, 0, 0
    
    def global_decoder(self, z, steps):
        out = torch.zeros((z.size(0), self.roll_dims)).cuda()
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = self.linear_init_global(z)
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        
        # if not self.training:
            # print("not training mode")

        for i in range(steps):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_g(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_g_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_g(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                # self.eps = self.k / \
                #     (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma, c_r_oh, c_n_oh,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis_r, dis_n = self.encoder(x, c_r_oh, c_n_oh, chroma, is_res=is_res)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)

        # # add tanh to regularized dimension constraint
        # z_r_new = torch.cat([torch.tanh(z_r[:, 0]).unsqueeze(-1), z_r[:, 1:]], dim=-1)
        # z_n_new = torch.cat([torch.tanh(z_n[:, 0]).unsqueeze(-1), z_n[:, 1:]], dim=-1)
        z_r_new = z_r
        z_n_new = z_n

        # get sub decoders output
        r_out, n_out, r_density, n_density = self.sub_decoders(rhythm, z_r_new, note, z_n_new)

        # packaging output
        z = torch.cat([z_r_new, z_n_new, chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        output = (out, r_out, n_out, r_density, n_density)
        dis = (dis_r, dis_n)
        z_out = (z_r_new, z_n_new)
        
        res = (output, dis, clas, z_out) if is_class else (output, dis, z_out)
        return res


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MusicAttrRegAdvVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 tempo_dims,
                 velocity_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrRegAdvVAE, self).__init__()
        
        # encoder
        self.gru_r = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_n = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_c = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)

        # sub-decoder
        self.gru_d_r = nn.GRU(z_dims + rhythm_dims, hidden_dims, batch_first=True)
        self.gru_d_n = nn.GRU(z_dims + note_dims, hidden_dims, batch_first=True)
        self.gru_d_c = nn.GRU(z_dims + chroma_dims, hidden_dims, batch_first=True)

        # classifiers
        self.c_r = nn.Linear(z_dims, 3)
        self.c_n = nn.Linear(z_dims, 3)
        
        # mu and logvar
        self.mu_r, self.var_r = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_n, self.var_n = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
        self.mu_c, self.var_c = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # global decoder
        num_dims = 2
        cdtl_dims = 24
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear init before sub-decoder
        self.linear_init_r = nn.Linear(z_dims, hidden_dims)
        self.linear_init_n = nn.Linear(z_dims, hidden_dims)
        self.linear_init_c = nn.Linear(z_dims, hidden_dims)

        # linear out after sub-decoder
        self.linear_out_r = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_out_n = nn.Linear(hidden_dims, note_dims)
        self.linear_out_c = nn.Linear(z_dims, chroma_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)

        # adversarial training layer
        self.r_linear_r = nn.Linear(z_dims-1, 1)
        self.r_linear_n = nn.Linear(z_dims-1, 1)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, c_r_oh, c_n_oh, chroma, is_res=False):
        # rhythm encoder
        x_r = self.gru_r(x)[-1]
        x_r = x_r.transpose_(0, 1).contiguous().view(x_r.size(0), -1)
        mu_r, var_r = self.mu_r(x_r), self.var_r(x_r).exp_()
        
        # note encoder
        x_n = self.gru_n(x)[-1]
        x_n = x_n.transpose_(0, 1).contiguous().view(x_n.size(0), -1)
        mu_n, var_n = self.mu_n(x_n), self.var_n(x_n).exp_()

        # chroma encoder
        # x_c = self.gru_c(x)[-1]
        # x_c = x_c.transpose_(0, 1).contiguous().view(x_c.size(0), -1)
        # mu_c, var_c = self.mu_c(x_c), self.var_c(x_c).exp_()

        dis_r = Normal(mu_r, var_r)
        dis_n = Normal(mu_n, var_n)
        # dis_c = Normal(mu_c, var_c)

        output = (dis_r, dis_n)

        return output

    def sub_decoders(self, rhythm, z_r, note, z_n):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        rhythm_in = get_hidden_and_concat_latent(rhythm, z_r)
        h_r = self.linear_init_r(z_r).unsqueeze(0)
        rhythm_out = self.gru_d_r(rhythm_in, h_r)[0]
        rhythm_out = F.log_softmax(self.linear_out_r(rhythm_out), 1)

        note_in = get_hidden_and_concat_latent(note, z_n)
        h_n = self.linear_init_n(z_n).unsqueeze(0)
        note_out = self.gru_d_n(note_in, h_n)[0]
        note_out = F.log_softmax(self.linear_out_n(note_out), 1)

        # chroma_out = torch.nn.Sigmoid()(self.linear_out_c(z_c))   # BCE

        # adversarial part
        r_z_r_rest = ReverseLayerF.apply(z_r[:, 1:])
        r_z_n_rest = ReverseLayerF.apply(z_n[:, 1:])

        r_r_density = self.r_linear_r(r_z_r_rest)
        r_n_density = self.r_linear_n(r_z_n_rest)

        return rhythm_out, note_out, r_r_density, r_n_density
    
    def global_decoder(self, z, steps):
        out = torch.zeros((z.size(0), self.roll_dims)).cuda()
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = self.linear_init_global(z)
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        
        # if not self.training:
            # print("not training mode")

        for i in range(steps):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_g(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_g_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_g(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                # self.eps = self.k / \
                #     (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma, c_r_oh, c_n_oh,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis_r, dis_n = self.encoder(x, c_r_oh, c_n_oh, chroma, is_res=is_res)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)

        # if is_class, get the classes for each latent
        if is_class:
            cls_r = self.c_r(z_r)
            cls_n = self.c_n(z_n)

        # get sub decoders output
        r_out, n_out, r_r_density, r_n_density = self.sub_decoders(rhythm, z_r, note, z_n)

        # packaging output
        z = torch.cat([z_r, z_n, chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        output = (out, r_out, n_out, r_r_density, r_n_density)
        dis = (dis_r, dis_n)
        z_out = (z_r, z_n)
        
        if is_class:
            clas = (cls_r, cls_n)
        
        res = (output, dis, clas, z_out) if is_class else (output, dis, z_out)
        return res


class MusicAttrNoteVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 tempo_dims,
                 velocity_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrNoteVAE, self).__init__()
        
        # encoder
        self.gru_n = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)

        # sub-decoder
        self.gru_d_n = nn.GRU(z_dims + note_dims, hidden_dims, batch_first=True)

        # classifiers
        self.c_r = nn.Linear(z_dims, 3)
        self.c_n = nn.Linear(z_dims, 3)
        
        # mu and logvar
        self.mu_n, self.var_n = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # global decoder
        num_dims = 1
        cdtl_dims = 0
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear init before sub-decoder
        self.linear_init_n = nn.Linear(z_dims, hidden_dims)

        # linear out after sub-decoder
        self.linear_out_n = nn.Linear(hidden_dims, note_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, c_r_oh, c_n_oh, chroma, is_res=False):
        
        # note encoder
        x_n = self.gru_n(x)[-1]
        x_n = x_n.transpose_(0, 1).contiguous().view(x_n.size(0), -1)
        mu_n, var_n = self.mu_n(x_n), self.var_n(x_n).exp_()

        # chroma encoder
        # x_c = self.gru_c(x)[-1]
        # x_c = x_c.transpose_(0, 1).contiguous().view(x_c.size(0), -1)
        # mu_c, var_c = self.mu_c(x_c), self.var_c(x_c).exp_()

        dis_n = Normal(mu_n, var_n)
        # dis_c = Normal(mu_c, var_c)

        output = dis_n

        return output

    def sub_decoders(self, note, z_n):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        note_in = get_hidden_and_concat_latent(note, z_n)
        h_n = self.linear_init_n(z_n).unsqueeze(0)
        note_out = self.gru_d_n(note_in, h_n)[0]
        note_out = F.log_softmax(self.linear_out_n(note_out), 1)

        # chroma_out = torch.nn.Sigmoid()(self.linear_out_c(z_c))   # BCE

        return note_out
    
    def global_decoder(self, z, steps):
        out = torch.zeros((z.size(0), self.roll_dims)).cuda()
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = self.linear_init_global(z)
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        
        # if not self.training:
            # print("not training mode")

        for i in range(steps):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_g(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_g_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_g(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                # self.eps = self.k / \
                #     (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma, c_r_oh, c_n_oh,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis_n = self.encoder(x, c_r_oh, c_n_oh, chroma, is_res=is_res)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_n = repar(dis_n.mean, dis_n.stddev)

        # if is_class, get the classes for each latent
        if is_class:
            cls_r = self.c_r(z_r)
            cls_n = self.c_n(z_n)

        # get sub decoders output
        n_out = self.sub_decoders(note, z_n)

        # packaging output
        out = self.global_decoder(z_n, steps=x.shape[1])
        output = (out, n_out)
        dis = dis_n
        z_out = z_n
        
        if is_class:
            clas = (cls_r, cls_n)
        
        res = (output, dis, clas, z_out) if is_class else (output, dis, z_out)
        return res


class MusicAttrCVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 tempo_dims,
                 velocity_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrCVAE, self).__init__()
        
        # encoder
        self.gru_e = nn.GRU(roll_dims + 2, hidden_dims, batch_first=True, bidirectional=True)

        # classifiers
        self.c_r = nn.Linear(z_dims, 3)
        self.c_n = nn.Linear(z_dims, 3)
        
        # mu and logvar
        self.mu, self.var = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # global decoder
        num_dims = 1
        cdtl_dims = 2
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear out after sub-decoder
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, r_density, n_density, chroma):
        r_density_rpt = torch.stack([r_density] * x.shape[1], dim=1)
        n_density_rpt = torch.stack([n_density] * x.shape[1], dim=1)
        x_in = torch.cat([x, r_density_rpt, n_density_rpt], dim=-1)

        # 1 encoder
        h = self.gru_e(x_in)[-1]
        h = h.transpose_(0, 1).contiguous().view(h.size(0), -1)
        mu, var = self.mu(h), self.var(h).exp_()

        dis = Normal(mu, var)

        return dis

    def sub_decoders(self, rhythm, z_r, note, z_n):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        rhythm_in = get_hidden_and_concat_latent(rhythm, z_r)
        h_r = self.linear_init_r(z_r).unsqueeze(0)
        rhythm_out = self.gru_d_r(rhythm_in, h_r)[0]
        rhythm_out = F.log_softmax(self.linear_out_r(rhythm_out), 1)

        note_in = get_hidden_and_concat_latent(note, z_n)
        h_n = self.linear_init_n(z_n).unsqueeze(0)
        note_out = self.gru_d_n(note_in, h_n)[0]
        note_out = F.log_softmax(self.linear_out_n(note_out), 1)

        # chroma_out = torch.nn.Sigmoid()(self.linear_out_c(z_c))   # BCE

        return rhythm_out, note_out, 0, 0
    
    def global_decoder(self, z, steps):
        out = torch.zeros((z.size(0), self.roll_dims)).cuda()
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = self.linear_init_global(z)
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        
        # if not self.training:
            # print("not training mode")

        for i in range(steps):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_g(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_g_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_g(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                # self.eps = self.k / \
                #     (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma, r_density, n_density,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis = self.encoder(x, r_density, n_density, chroma)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)

        # packaging output
        z = torch.cat([z, r_density, n_density], dim=-1)     
        out = self.global_decoder(z, steps=x.shape[1])

        res = (out, dis, z)
        return res


class MusicAttrTransformerVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrTransformerVAE, self).__init__()
        
        # encoder
        self.embedding = nn.Embedding(roll_dims, hidden_dims)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dims, 
                                                        nhead=4, 
                                                        dim_feedforward=hidden_dims, 
                                                        dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                        num_layers=6)
        
        # mu and logvar
        self.mu, self.var = nn.Linear(hidden_dims * n_step, z_dims), nn.Linear(hidden_dims * n_step, z_dims)
    
        # global decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dims, 
                                                        nhead=4)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, 
                                                        num_layers=6)

        # linear init before sub-decoder
        self.linear_init = nn.Linear(z_dims, hidden_dims * n_step)

        # linear out after sub-decoder
        self.linear_out = nn.Linear(hidden_dims, roll_dims)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x):
        x = self.embedding(x)
        h = self.transformer_encoder(x)

        h = h.view(h.size(0), h.size(1) * h.size(2))

        mu, var = self.mu(h), self.var(h)
        return Normal(mu, var)
    
    def decoder(self, z, x):
        pad_zeros = torch.zeros(x.size(0)).cuda().unsqueeze(-1).long()
        x_new = torch.cat([pad_zeros, x[:, :-1]], dim=1)

        h = self.linear_init(z)
        h = h.view(z.size(0), x.size(1), int(h.size(1) / x.size(1)))

        x_oh = self.embedding(x_new)
        out = self.transformer_decoder(x_oh, h)
        out = self.linear_out(out)
        return out

    def forward(self, x):
        
        dis = self.encoder(x)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)
    
        out = self.decoder(z, x)
        return out, dis


class MusicGenericAttrRegVAE(nn.Module):
    def __init__(self,
                 roll_dims,
                 attr_dims_lst,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        
        super(MusicGenericAttrRegVAE, self).__init__()

        # initialization
        num_attrs = len(attr_dims_lst)
        
        # encoder
        self.gru_encoder_lst = torch.nn.ModuleList()
        for _ in range(num_attrs):
            self.gru_encoder_lst.append(nn.GRU(roll_dims, hidden_dims, batch_first=True, 
                                                bidirectional=True))

        # sub-decoder
        self.gru_sub_decoder_lst = torch.nn.ModuleList()
        for i in range(num_attrs):
            attr_dims = attr_dims_lst[i]
            self.gru_sub_decoder_lst.append(nn.GRU(z_dims + attr_dims, hidden_dims, batch_first=True))

        # mu and logvar
        self.mu_lst= torch.nn.ModuleList()
        for _ in range(num_attrs):
            self.mu_lst.append(nn.Linear(hidden_dims * 2, z_dims))

        self.var_lst= torch.nn.ModuleList()
        for _ in range(num_attrs):
            self.var_lst.append(nn.Linear(hidden_dims * 2, z_dims))
    
        # global decoder
        num_dims = len(attr_dims_lst) - 1       # exclude harmony vector
        cdtl_dims = 24                          # harmony vector
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear init before sub-decoder
        self.linear_init_lst = torch.nn.ModuleList()
        for _ in range(num_attrs):
            self.linear_init_lst.append(nn.Linear(z_dims, hidden_dims))

        # linear out after sub-decoder
        self.linear_out_lst = torch.nn.ModuleList()
        for i in range(num_attrs):
            self.linear_out_lst.append(nn.Linear(hidden_dims, attr_dims_lst[i]))

        # global decoder final linear layer
        self.linear_init_g = nn.Linear(z_dims, hidden_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)
        
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 100
        self.sample = None
        self.iteration = 0
        self.z_dims = z_dims
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, attr_lst, is_res=False):
        dis_lst = []
        
        for i in range(len(attr_lst) - 1):
            # exclude chroma, not encoded
            attr = attr_lst[i]
            x_attr = self.gru_encoder_lst[i](x)[-1]
            x_attr = x_attr.transpose_(0, 1).contiguous().view(x_attr.size(0), -1)
            mu_attr, var_attr = self.mu_lst[i](x_attr), self.var_lst[i](x_attr).exp_()
            dis_lst.append(Normal(mu_attr, var_attr))

        return dis_lst

    def sub_decoders(self, attr_lst, z_lst):

        def get_hidden_and_concat_latent(input, z_latent):
            z_latent_stack = torch.stack([z_latent] * input.shape[1], dim=1)
            input_in = torch.cat([input, z_latent_stack], dim=-1)
            return input_in

        output_lst = []
        
        for i in range(len(z_lst)):
            attr, z_attr = attr_lst[i], z_lst[i]
            attr_in = get_hidden_and_concat_latent(attr, z_attr)
            h_attr = self.linear_init_lst[i](z_attr).unsqueeze(0)
            attr_out = self.gru_sub_decoder_lst[i](attr_in, h_attr)[0]
            attr_out = F.log_softmax(self.linear_out_lst[i](attr_out), 1)
            output_lst.append(attr_out)

        return output_lst
    
    def global_decoder(self, z, steps):
        out = torch.zeros((z.size(0), self.roll_dims)).cuda()
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = self.linear_init_global(z)
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()

        for i in range(steps):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_g(out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_g_2(hx[0], hx[1])
            out = F.log_softmax(self.linear_out_g(hx[1]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                # self.eps = self.k / \
                #     (self.k + torch.exp(self.iteration / self.k))
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, attr_lst, is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis_lst = self.encoder(x, attr_lst, is_res=is_res)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_lst = []
        for dis in dis_lst:
            z_lst.append(repar(dis.mean, dis.stddev))

        # get sub decoders output
        sub_output_lst = self.sub_decoders(attr_lst, z_lst)

        chroma = attr_lst[-1]

        # packaging output
        z = torch.cat(z_lst + [chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        dis = dis_lst
        z_out = z_lst
        
        res = (out, dis, z_out, sub_output_lst)
        return res

