import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.autograd import Function
from collections import Counter


class MusicAttrRegVAE(nn.Module):
    '''
    Music FaderNets, vanilla VAE model.
    Regularization loss can be GLSR or Pati et al. in trainer section.
    '''
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
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

    def encoder(self, x):
        # rhythm encoder
        x_r = self.gru_r(x)[-1]
        x_r = x_r.transpose_(0, 1).contiguous().view(x_r.size(0), -1)
        mu_r, var_r = self.mu_r(x_r), self.var_r(x_r).exp_()
        
        # note encoder
        x_n = self.gru_n(x)[-1]
        x_n = x_n.transpose_(0, 1).contiguous().view(x_n.size(0), -1)
        mu_n, var_n = self.mu_n(x_n), self.var_n(x_n).exp_()

        dis_r = Normal(mu_r, var_r)
        dis_n = Normal(mu_n, var_n)

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

        return rhythm_out, note_out
    
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
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma):
        if self.training:
            self.sample = x
            self.iteration += 1
        
        dis_r, dis_n = self.encoder(x)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)

        # get sub decoders output
        r_out, n_out = self.sub_decoders(rhythm, z_r, note, z_n)

        # packaging output
        z = torch.cat([z_r, z_n, chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        output = (out, r_out, n_out)
        dis = (dis_r, dis_n)
        z_out = (z_r, z_n)
        
        res = (output, dis, z_out)
        return res


class MusicAttrSingleVAE(nn.Module):
    '''
    Single encoder VAE with reg. loss by Pati et al. (2019).
    '''
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrSingleVAE, self).__init__()
        
        # encoder
        self.gru = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)

        # dropouts
        self.e_dropout = nn.Dropout(p=0.3)

        # no sub-decoder -- only latent regularization in loss function
        
        # mu and logvar -- use 2 * z_dims to ensure same capacity with disentangled models
        self.mu, self.var = nn.Linear(hidden_dims * 2, z_dims * 2), nn.Linear(hidden_dims * 2, z_dims * 2)
    
        # global decoder
        num_dims = 2
        cdtl_dims = 24
        self.linear_init_global = nn.Linear(z_dims * num_dims + cdtl_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims * num_dims + cdtl_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)
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

    def encoder(self, x):
        # encoder
        x = self.gru(x)[-1]
        x = x.transpose_(0, 1).contiguous().view(x.size(0), -1)
        mu, var = self.mu(x), self.var(x).exp_()

        return Normal(mu, var)

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

    def forward(self, x, chroma):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis = self.encoder(x)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)

        # packaging output
        z = torch.cat([z, chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        
        res = (out, dis, z)
        return res


class MusicAttrCVAE(nn.Module):
    '''
    CVAE model - one encoder, decode with concatenated conditions.
    '''
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
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

        return rhythm_out, note_out, 0, 0
    
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
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x, rhythm, note, chroma, r_density, n_density):
        
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

    
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

    
class MusicAttrFaderNets(nn.Module):
    '''
    Fader Networks model - basically a CVAE with adversarial loss training.
    '''
    def __init__(self,
                 roll_dims,
                 rhythm_dims,
                 note_dims,
                 chroma_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 k=1000):
        super(MusicAttrFaderNets, self).__init__()
        
        # encoder
        self.gru_e = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)

        # classifiers
        self.c_r = nn.Linear(z_dims, 3)
        self.c_n = nn.Linear(z_dims, 3)
        
        # mu and logvar
        self.mu, self.var = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)

        # discriminator
        self.discriminator_r = nn.Linear(z_dims, 1)
        self.discriminator_n = nn.Linear(z_dims, 1)
        self.dropout = nn.Dropout(p=0.3)
    
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

    def encoder(self, x):
        h = self.gru_e(x)[-1]
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

    def forward(self, x, rhythm, note, chroma, r_density, n_density):
        
        if self.training:
            self.sample = x
            self.iteration += 1
        
        # residual or without
        dis = self.encoder(x)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)

        # discriminator part
        r_z = ReverseLayerF.apply(z)
        r_out = self.dropout(F.relu(self.discriminator_r(r_z)))
        n_out = self.dropout(F.relu(self.discriminator_n(r_z)))

        # packaging output
        z = torch.cat([z, r_density, n_density], dim=-1)     
        out = self.global_decoder(z, steps=x.shape[1])
        output = (out, r_out, n_out)

        res = (output, dis, z)
        return res

