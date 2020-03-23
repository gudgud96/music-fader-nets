import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.autograd import Function
import numpy as np
from collections import Counter


class MusicAttrRegGMVAE(nn.Module):
    """
    MusicAttrVAE with a GMM as latent prior distribution.
    Reference: https://github.com/yjlolo/vae-audio/blob/master/model/model.py
    """
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
                 n_component=4):

        super(MusicAttrRegGMVAE, self).__init__()

        self.n_component = n_component
        self.latent_dim = z_dims
        self.roll_dims = roll_dims
        self.eps = 100
        
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

        # build latent mean and variance lookup
        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-3)       # a hyperparameter to set
    
    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encode(self, x):
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

    def _build_mu_lookup(self):
        """
        Follow Xavier initialization as in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        This can also be done using a GMM on the latent space trained with vanilla autoencoders,
        as in https://arxiv.org/abs/1611.05148.
        """
        mu_r_lookup = nn.Embedding(self.n_component, self.latent_dim)
        nn.init.xavier_uniform_(mu_r_lookup.weight)
        mu_r_lookup.weight.requires_grad = True
        self.mu_r_lookup = mu_r_lookup

        mu_n_lookup = nn.Embedding(self.n_component, self.latent_dim)
        nn.init.xavier_uniform_(mu_n_lookup.weight)
        mu_n_lookup.weight.requires_grad = True
        self.mu_n_lookup = mu_n_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        """
        Follow Table 7 in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        """
        logvar_r_lookup = nn.Embedding(self.n_component, self.latent_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_r_lookup.weight, init_logvar)
        logvar_r_lookup.weight.requires_grad = logvar_trainable
        self.logvar_r_lookup = logvar_r_lookup

        logvar_n_lookup = nn.Embedding(self.n_component, self.latent_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_n_lookup.weight, init_logvar)
        logvar_n_lookup.weight.requires_grad = logvar_trainable
        self.logvar_n_lookup = logvar_n_lookup
        # self.logvar_bound = np.log(np.exp(-1) ** 2)  # lower bound of log variance for numerical stability

    def _bound_logvar_lookup(self):
        self.logvar_lookup.weight.data[torch.le(self.logvar_lookup.weight, self.logvar_bound)] = self.logvar_bound

    def _infer_class(self, q_z, ):
        logLogit_qy_x, qy_x = self._approx_qy_x(q_z, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
        val, y = torch.max(qy_x, dim=1)
        return logLogit_qy_x, qy_x, y

    def approx_qy_x(self, z, mu_lookup, logvar_lookup, n_component):
        """
        Refer to eq.13 in the paper https://openreview.net/pdf?id=rygkk305YQ.
        Approximating q(y|x) with p(y|z), the probability of z being assigned to class y.
        q(y|x) ~= p(y|z) = p(z|y)p(y) / p(z)
        :param z: latent variables sampled from approximated posterior q(z|x)
        :param mu_lookup: i-th row corresponds to a mean vector of p(z|y = i) which is a Gaussian
        :param logvar_lookup: i-th row corresponds to a logvar vector of p(z|y = i) which is a Gaussian
        :param n_component: number of components of the GMM prior
        """
        def log_gauss_lh(z, mu, logvar):
            """
            Calculate p(z|y), the likelihood of z w.r.t. a Gaussian component
            """
            llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
            llh = torch.sum(llh, dim=1)  # sum over dimensions
            return llh

        logLogit_qy_x = torch.zeros(z.shape[0], n_component).cuda()  # log-logit of q(y|x)
        for k_i in torch.arange(0, n_component):
            mu_k, logvar_k = mu_lookup(k_i.cuda()), logvar_lookup(k_i.cuda())
            logLogit_qy_x[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

        qy_x = torch.nn.functional.softmax(logLogit_qy_x, dim=1)
        return logLogit_qy_x, qy_x

    def forward(self, x, rhythm, note, chroma, c_r_oh, c_n_oh,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
        
        # ========================== INFERENCE ====================== #
        # infer latent
        dis_r, dis_n = self.encode(x)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)

        # infer gaussian component
        logLogit_qy_x_r, qy_x_r = self.approx_qy_x(z_r, self.mu_r_lookup, self.logvar_r_lookup, n_component=self.n_component)
        _, y_r = torch.max(qy_x_r, dim=1)

        logLogit_qy_x_n, qy_x_n = self.approx_qy_x(z_n, self.mu_n_lookup, self.logvar_n_lookup, n_component=self.n_component)
        _, y_n = torch.max(qy_x_n, dim=1)

         # ========================== GENERATION ====================== #
        # get sub decoders output
        r_out, n_out, r_density, n_density = self.sub_decoders(rhythm, z_r, note, z_n)

        # packaging output
        z = torch.cat([z_r, z_n, chroma], dim=1)        
        out = self.global_decoder(z, steps=x.shape[1])
        output = (out, r_out, n_out, r_density, n_density)
        dis = (dis_r, dis_n)
        z_out = (z_r, z_n)
        qy_x_out = (qy_x_r, qy_x_n)
        logLogit_out = (logLogit_qy_x_r, logLogit_qy_x_n)
        y_out = (y_r, y_n)

        res = (output, dis, z_out, logLogit_out, qy_x_out, y_out)
        return res


class MusicAttrSingleGMVAE(nn.Module):
    """
    MusicAttrVAE with a GMM as latent prior distribution, without attribute modelling.
    Reference: https://github.com/yjlolo/vae-audio/blob/master/model/model.py
    """
    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 z_dims,
                 n_step,
                 n_component=4):

        super(MusicAttrSingleGMVAE, self).__init__()

        self.n_component = n_component
        self.latent_dim = z_dims
        self.roll_dims = roll_dims
        self.eps = 100
        
        # encoder
        self.gru = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)

        # dropouts
        self.e_dropout = nn.Dropout(p=0.3)
        
        # mu and logvar
        self.mu, self.var = nn.Linear(hidden_dims * 2, z_dims), nn.Linear(hidden_dims * 2, z_dims)
    
        # global decoder
        num_dims = 2
        cdtl_dims = 24
        self.linear_init_global = nn.Linear(z_dims, hidden_dims)
        self.grucell_g = nn.GRUCell(z_dims + roll_dims, hidden_dims)
        self.grucell_g_2 = nn.GRUCell(hidden_dims, hidden_dims)

        # linear init before sub-decoder
        self.linear_init = nn.Linear(z_dims, hidden_dims)
        self.linear_out_g = nn.Linear(hidden_dims, roll_dims)

        # build latent mean and variance lookup
        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)       # a hyperparameter to set
    
    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encode(self, x):
        # rhythm encoder
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

    def _build_mu_lookup(self):
        """
        Follow Xavier initialization as in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        This can also be done using a GMM on the latent space trained with vanilla autoencoders,
        as in https://arxiv.org/abs/1611.05148.
        """
        mu_lookup = nn.Embedding(self.n_component, self.latent_dim)
        nn.init.xavier_uniform_(mu_lookup.weight)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        """
        Follow Table 7 in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        """
        logvar_lookup = nn.Embedding(self.n_component, self.latent_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup
        # self.logvar_bound = np.log(np.exp(-1) ** 2)  # lower bound of log variance for numerical stability

    def _bound_logvar_lookup(self):
        self.logvar_lookup.weight.data[torch.le(self.logvar_lookup.weight, self.logvar_bound)] = self.logvar_bound

    def _infer_class(self, q_z, ):
        logLogit_qy_x, qy_x = self._approx_qy_x(q_z, self.mu_lookup, self.logvar_lookup, n_component=self.n_component)
        val, y = torch.max(qy_x, dim=1)
        return logLogit_qy_x, qy_x, y

    def approx_qy_x(self, z, mu_lookup, logvar_lookup, n_component):
        """
        Refer to eq.13 in the paper https://openreview.net/pdf?id=rygkk305YQ.
        Approximating q(y|x) with p(y|z), the probability of z being assigned to class y.
        q(y|x) ~= p(y|z) = p(z|y)p(y) / p(z)
        :param z: latent variables sampled from approximated posterior q(z|x)
        :param mu_lookup: i-th row corresponds to a mean vector of p(z|y = i) which is a Gaussian
        :param logvar_lookup: i-th row corresponds to a logvar vector of p(z|y = i) which is a Gaussian
        :param n_component: number of components of the GMM prior
        """
        def log_gauss_lh(z, mu, logvar):
            """
            Calculate p(z|y), the likelihood of z w.r.t. a Gaussian component
            """
            llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
            llh = torch.sum(llh, dim=1)  # sum over dimensions
            return llh

        logLogit_qy_x = torch.zeros(z.shape[0], n_component).cuda()  # log-logit of q(y|x)
        for k_i in torch.arange(0, n_component):
            mu_k, logvar_k = mu_lookup(k_i.cuda()), logvar_lookup(k_i.cuda())
            logLogit_qy_x[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

        qy_x = torch.nn.functional.softmax(logLogit_qy_x, dim=1)
        return logLogit_qy_x, qy_x

    def forward(self, x, rhythm, note, chroma, c_r_oh, c_n_oh,
                is_class=False, is_res=False):
        
        if self.training:
            self.sample = x
        
        # ========================== INFERENCE ====================== #
        # infer latent
        dis = self.encode(x)
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)

        # infer gaussian component
        logLogit_qy_x, qy_x = self.approx_qy_x(z, self.mu_r_lookup, self.logvar_r_lookup, n_component=self.n_component)
        _, y = torch.max(qy_x, dim=1)

         # ========================== GENERATION ====================== #
        # packaging output

        out = self.global_decoder(z, steps=x.shape[1])
        output = out
        dis = dis
        z_out = z
        qy_x_out = qy_x
        logLogit_out = logLogit_qy_x
        y_out = y

        res = (output, dis, z_out, logLogit_out, qy_x_out, y_out)
        return res


