from dataclasses import dataclass
from torch import nn
import torch
import torch.nn.functional as F



@dataclass
class GWNETConfig:
     num_nodes: int = 883
     dropout: float = 0.5
     gcn_bool: bool = True
     addaptadj: bool = True
     in_dim: int = 1
     out_dim: int = 12
     residual_channels: int = 64
     dilation_channels: int = 64
     skip_channels: int = 256
     end_channels: int = 512
     kernel_size: int = 3
     blocks: int = 2
     layers: int = 4
     apt_size: int = 64
     alpha:float = 0.01
     nheads:int = 4


@dataclass
class ModGWNETConfig:
     num_nodes: int = 358
     dropout: float = 0.5
     gcn_bool: bool = True
     addaptadj: bool = True
     in_dim: int = 1
     use_tcn: bool = True
     use_gconv: bool = True
     use_gate: bool = True
     use_stam: bool = True
     use_gat: bool = True
     out_dim: int = 12
     residual_channels: int = 64
     dilation_channels: int = 64
     skip_channels: int = 256
     end_channels: int = 512
     kernel_size: int = 3
     blocks: int = 2
     layers: int = 4
     apt_size: int = 64
     alpha:float = 0.01
     nheads:int = 4
     tcn_drop: float = 0.0
     stam_drop: float = 0.0
     

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0]
    

class TCN(nn.Module):
    def __init__(
        self,
        residual_channels=32,
        dilation_channels=32,
        kernel_size=3,
        drop=0.0,
        dilation=1,
        activation="ReLU",
    ):
        super(TCN, self).__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.padding = nn.ConstantPad2d((padding, padding, 0, 0), 0)
        self.conv = nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels * 2,
                                               kernel_size=(1, kernel_size), dilation=dilation)
        self.act_fn = getattr(nn, activation)()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, input):
        x = self.padding(input)
        x, gate = self.conv(x).chunk(2, dim=1)
        x = x * torch.sigmoid(gate)
        x = self.drop(x)
        return x


class GatedTCN(nn.Module):
    def __init__(
        self,
        residual_channels=32,
        dilation_channels=32,
        kernel_size=3,
        nlayers=2,
        drop=0.0,
        activation="ReLU",
    ):
        super(GatedTCN, self).__init__()
        self.convs = nn.ModuleList()

        new_dilation = 1
        for _ in range(nlayers):
            # dilated convolutions
            self.convs.append(
                TCN(
                    residual_channels=residual_channels,
                    dilation_channels=dilation_channels,
                    kernel_size=kernel_size,
                    drop=drop,
                    dilation=new_dilation,
                    activation=activation,
                )
            )


            new_dilation *= 2
    def forward(self, input):
        x = input
        for tcn in self.convs:
            x = tcn(x)
        return x
    

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(GraphConv, self).__init__()
        c_in = (order * support_len + 1) * c_in
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def nconv(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


    def forward(self, x, support):
        out = [x]
        for adj_mat in support:
            x1 = self.nconv(x, adj_mat)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, adj_mat)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha=0.1, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -torch.finfo(e.dtype).max * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = nn.ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x
    

class Gate(nn.Module):
    def __init__(self, n_out):
        super(Gate, self).__init__()
        self.n_out = n_out

        self.w_gate1 = nn.Linear(n_out, n_out)
        self.w_gate2 = nn.Linear(n_out, n_out)
        
    def forward(self, x, h):
        gate = torch.sigmoid(self.w_gate1(x) + self.w_gate2(h))
        one_vec = torch.ones_like(gate)
        z = gate * x + (one_vec - gate) * h
        return z
    
class AHSTLBlock(nn.Module):
  def __init__(self, config, supports=None, support_len=1):
      super(AHSTLBlock, self).__init__()
      self.supports = supports
      self.gcn_bool = config.gcn_bool
      self.addaptadj = config.addaptadj
      self.skip_conv = nn.Conv2d(
          in_channels=config.dilation_channels,
          out_channels=config.skip_channels,
          kernel_size=(1, 1)
      )
      self.tcr = GatedTCN(
          residual_channels=config.residual_channels,
          dilation_channels=config.dilation_channels,
          kernel_size=config.kernel_size,
          nlayers=config.layers,
          drop=config.tcn_drop

      ) if config.use_tcn else IdentityLayer()

      self.tcd = GatedTCN(
          residual_channels=config.residual_channels,
          dilation_channels=config.dilation_channels,
          kernel_size=config.kernel_size,
          nlayers=config.layers,
          drop=config.tcn_drop
      ) if config.use_tcn else IdentityLayer()

      self.tcw = GatedTCN(
          residual_channels=config.residual_channels,
          dilation_channels=config.dilation_channels,
          kernel_size=config.kernel_size,
          nlayers=config.layers,
          drop=config.tcn_drop
      ) if config.use_tcn else IdentityLayer()

      self.mlp = nn.Conv2d(in_channels=config.residual_channels * 3,
                    out_channels=config.residual_channels,
                    kernel_size=(1, 1))

      self.bn = nn.BatchNorm2d(config.residual_channels)
      if config.use_gconv:
        self.gconv = GraphConv(
            config.dilation_channels,
            config.residual_channels,
            config.dropout,
            support_len=support_len,
        )
      else:
        self.gconv = IdentityLayer()

      self.w_t = nn.Conv2d(in_channels=config.apt_size,
                            out_channels=config.residual_channels,
                            kernel_size=(1, 1))
      self.w_ls = nn.Conv2d(in_channels=config.apt_size,
                             out_channels=config.residual_channels,
                             kernel_size=(1, 1))
      self.gat = GAT(
          config.residual_channels,
          config.residual_channels,
          config.dropout,
          config.alpha,
          config.nheads,
      ) if config.use_gat else IdentityLayer()
      self.gate = Gate(config.residual_channels) if config.use_gate else IdentityLayer()
      self.use_stam = config.use_stam
      self.stam_drop = None
      if config.stam_drop > 0.0:
        self.stam_drop = nn.Dropout(config.stam_drop)

  def compute_attention(self,  x, node_vec):
      n_q = node_vec.unsqueeze(dim=-1)
      x_t_a = torch.einsum('btnd,ndl->btnl', (x, n_q)).contiguous()
      return x_t_a

  def compute_time_attn(self, x, x_t, nodevec1, new_supports=None):
      n_q_t = self.w_t(nodevec1.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()
      x_t_a = self.compute_attention(x_t, n_q_t)
      if self.gcn_bool and self.supports is not None:
          if self.addaptadj:
              x = self.gconv(x, new_supports)
          else:
              x = self.gconv(x, self.supports)
      return x, x_t_a

  def compute_long_attn(self, x_t, x_ls, nodevec1):
      # Add GAT
      x_ds = self.gat(x_t, self.supports[0])
      x_ls = self.gate(x_ls, x_ds)

      # Add Long/Static Spatial attention
      # compute Long/Static Spatial attention score
      n_q_ls = self.w_ls(nodevec1.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()
      x_ls_a = self.compute_attention(x_ls, n_q_ls)
      return x_ls_a


  def forward(
      self,
      rec,
      day,
      week,
      nodevec1,
      skip=None,
      new_supports=None
  ):

      x1 =  self.tcr(rec)
      x2 = self.tcd(day)
      x3 = self.tcw(week)
      x = self.mlp(torch.cat([x1, x2, x3], dim=1))

      x_t = x.transpose(1, 3)
      x, x_t_a = self.compute_time_attn(x, x_t, nodevec1, new_supports)
      x = self.bn(x)

      # Long/Static Spatial feature x_ls
      x_ls = x.transpose(1, 3)
      x_ls_a = self.compute_long_attn(x_t, x_ls, nodevec1)

      # node-level adaptation tendencies
      if self.use_stam:
        x_a = torch.cat((x_t_a, x_ls_a), -1)
        x_att = F.softmax(x_a, dim=-1)
        if self.stam_drop is not None:
          x_att = self.stam_drop(x_att)
        # Stam Layer
        x = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_t + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_ls
      else:
        x = x_t + x_ls
      x = x.transpose(1, 3)
      # parametrized skip connection
      s = x
      s = self.skip_conv(s)
      if skip is not None:
        skip = skip[:, :, :, -s.size(3):]
        skip = s + skip

      return s

class GWNET(nn.Module):
    def __init__(self, config, supports=None, aptinit=None):
        super(GWNET, self).__init__()
        self.config = config
        self.dropout = config.dropout
        self.layers = config.layers
        self.gcn_bool = config.gcn_bool
        self.addaptadj = config.addaptadj
        self.kernel_size = config.kernel_size

        skip_channels = config.skip_channels
        end_channels = config.end_channels
        dilation_channels = config.dilation_channels
        residual_channels = config.residual_channels
        out_dim = config.out_dim

        self.start_rec = nn.Conv2d(in_channels=config.in_dim,
                                    out_channels=config.residual_channels,
                                    kernel_size=(1, 1)
                                    )
        self.start_day = nn.Conv2d(in_channels=config.in_dim,
                                    out_channels=config.residual_channels,
                                    kernel_size=(1, 1)
                                    )

        self.start_week = nn.Conv2d(in_channels=config.in_dim,
                                    out_channels=config.residual_channels,
                                    kernel_size=(1, 1)
                                    )
        self.supports = supports

        receptive_field = 1
        for _ in range(config.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for _ in range(config.layers):
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if self.gcn_bool and self.addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(config.num_nodes, config.apt_size), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(config.apt_size, config.num_nodes), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        if self.gcn_bool and self.addaptadj and self.supports is not None:
           self.supports_len = 1
        self.blocks = nn.ModuleList([AHSTLBlock(config, supports, self.supports_len) for _ in range(config.blocks)])
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, rec, day, week):
        # Start Conv
        rec = self.start_rec(rec)
        day = self.start_day(day)
        week = self.start_week(week)

        skip = None

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            N = self.supports[0].shape[0]
            dev = self.nodevec1.device
            adp = torch.eye(N).to(dev) + F.relu(torch.tanh(torch.mm(self.nodevec1, self.nodevec2)))
            adp =  torch.softmax(adp, dim=1)
            new_supports = [adp]


        # WaveNet Blocks
        for block in self.blocks:
            nodevec = self.nodevec1 if self.addaptadj else None
            #print(skip.shape if skip is  not None else "")
            skip = block(rec, day, week, nodevec, skip=skip, new_supports=new_supports)


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    

def get_model(adj, **kwargs):
    config = ModGWNETConfig(**kwargs)
    return GWNET(config, supports=adj)
