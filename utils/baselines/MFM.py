import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from utils.models_transformer import BertTextEncoder


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def loss_MMD(zy):
    zy_real_gauss = Variable(torch.randn(zy.size()))  # no need to be the same size

    # if args.cuda:
    zy_real_gauss = zy_real_gauss.cuda()
    zy_real_kernel = compute_kernel(zy_real_gauss, zy_real_gauss)
    zy_fake_kernel = compute_kernel(zy, zy)
    zy_kernel = compute_kernel(zy_real_gauss, zy)
    zy_mmd = zy_real_kernel.mean() + zy_fake_kernel.mean() - 2.0 * zy_kernel.mean()
    return zy_mmd


class encoderLSTM(nn.Module):
    def __init__(self, d, h):  # , n_layers, bidirectional, dropout):
        super(encoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(d, h)
        self.fc1 = nn.Linear(h, h)
        self.h = h

    def forward(self, x):
        # x is t x n x h
        t = x.shape[0]
        n = x.shape[1]
        self.hx = torch.zeros(n, self.h).cuda()
        self.cx = torch.zeros(n, self.h).cuda()
        all_hs = []
        all_cs = []
        for i in range(t):
            self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
        # last hidden layer last_hs is n x h
        last_hs = all_hs[-1]
        last_hs = self.fc1(last_hs)
        return last_hs


class decoderLSTM(nn.Module):
    def __init__(self, h, d):
        super(decoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(h, h)
        self.fc1 = nn.Linear(h, d)
        self.d = d
        self.h = h

    def forward(self, hT, t):  # only embedding vector
        # x is n x d
        n = hT.shape[0]
        h = hT.shape[1]
        self.hx = torch.zeros(n, self.h).cuda()
        self.cx = torch.zeros(n, self.h).cuda()
        final_hs = []
        all_hs = []
        all_cs = []
        for i in range(t):
            if i == 0:
                self.hx, self.cx = self.lstm(hT, (self.hx, self.cx))
            else:
                self.hx, self.cx = self.lstm(all_hs[-1], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
            final_hs.append(self.hx.view(1, n, h))
        final_hs = torch.cat(final_hs, dim=0)
        all_recons = self.fc1(final_hs)
        return all_recons


class MFN(nn.Module):
    def __init__(self, config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig):
        super(MFN, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = config["h_dims"]
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 1
        attInShape = total_h_dim * window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]
        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x):
        x_l = x[:, :, :self.d_l]
        x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        x_v = x[:, :, self.d_l + self.d_a:]
        # x is t x n x d
        n = x.shape[1]
        t = x.shape[0]
        self.h_l = torch.zeros(n, self.dh_l).cuda()
        self.h_a = torch.zeros(n, self.dh_a).cuda()
        self.h_v = torch.zeros(n, self.dh_v).cuda()
        self.c_l = torch.zeros(n, self.dh_l).cuda()
        self.c_a = torch.zeros(n, self.dh_a).cuda()
        self.c_v = torch.zeros(n, self.dh_v).cuda()
        self.mem = torch.zeros(n, self.mem_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l, prev_c_a, prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)
            cStar = torch.cat([prev_cs, new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))), dim=1)
            attended = attention * cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended, self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1 * self.mem + gamma2 * cHat
            all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)
        return last_hs


class MFM(nn.Module):
    def __init__(self, config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig):
        super(MFM, self).__init__()

        self.text_model = BertTextEncoder()  # BERT version

        [self.d_l, self.d_a, self.d_v] = config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = config["h_dims"]
        zy_size = config['zy_size']
        zl_size = config['zl_size']
        za_size = config['za_size']
        zv_size = config['zv_size']
        fy_size = config['fy_size']
        fl_size = config['fl_size']
        fa_size = config['fa_size']
        fv_size = config['fv_size']
        zy_to_fy_dropout = config['zy_to_fy_dropout']
        zl_to_fl_dropout = config['zl_to_fl_dropout']
        za_to_fa_dropout = config['za_to_fa_dropout']
        zv_to_fv_dropout = config['zv_to_fv_dropout']
        fy_to_y_dropout = config['fy_to_y_dropout']
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        last_mfn_size = total_h_dim + config["memsize"]
        output_dim = 1

        self.encoder_l = encoderLSTM(self.d_l, zl_size)
        self.encoder_a = encoderLSTM(self.d_a, za_size)
        self.encoder_v = encoderLSTM(self.d_v, zv_size)

        self.decoder_l = decoderLSTM(fy_size + fl_size, self.d_l)
        self.decoder_a = decoderLSTM(fy_size + fa_size, self.d_a)
        self.decoder_v = decoderLSTM(fy_size + fv_size, self.d_v)

        self.mfn_encoder = MFN(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig)
        self.last_to_zy_fc1 = nn.Linear(last_mfn_size, zy_size)

        self.zy_to_fy_fc1 = nn.Linear(zy_size, fy_size)
        self.zy_to_fy_fc2 = nn.Linear(fy_size, fy_size)
        self.zy_to_fy_dropout = nn.Dropout(zy_to_fy_dropout)

        self.zl_to_fl_fc1 = nn.Linear(zl_size, fl_size)
        self.zl_to_fl_fc2 = nn.Linear(fl_size, fl_size)
        self.zl_to_fl_dropout = nn.Dropout(zl_to_fl_dropout)

        self.za_to_fa_fc1 = nn.Linear(za_size, fa_size)
        self.za_to_fa_fc2 = nn.Linear(fa_size, fa_size)
        self.za_to_fa_dropout = nn.Dropout(za_to_fa_dropout)

        self.zv_to_fv_fc1 = nn.Linear(zv_size, fv_size)
        self.zv_to_fv_fc2 = nn.Linear(fv_size, fv_size)
        self.zv_to_fv_dropout = nn.Dropout(zv_to_fv_dropout)

        self.fy_to_y_fc1 = nn.Linear(fy_size, fy_size)
        self.fy_to_y_fc2 = nn.Linear(fy_size, output_dim)
        self.fy_to_y_dropout = nn.Dropout(fy_to_y_dropout)

    def forward(self, x_l, x_a, x_v):
        # x_l = x[:, :, :self.d_l]
        # x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        # x_v = x[:, :, self.d_l + self.d_a:]
        x_l = self.text_model(x_l).transpose(0, 1)
        x = torch.cat((x_l, x_a, x_v), dim=-1)
        # x is t x n x d
        n = x.shape[1]
        t = x.shape[0]

        zl = self.encoder_l.forward(x_l)
        za = self.encoder_a.forward(x_a)
        zv = self.encoder_v.forward(x_v)

        mfn_last = self.mfn_encoder.forward(x)
        zy = self.last_to_zy_fc1(mfn_last)
        mmd_loss = loss_MMD(zl) + loss_MMD(za) + loss_MMD(zv) + loss_MMD(zy)
        missing_loss = 0.0

        fy = F.relu(self.zy_to_fy_fc2(self.zy_to_fy_dropout(F.relu(self.zy_to_fy_fc1(zy)))))
        fl = F.relu(self.zl_to_fl_fc2(self.zl_to_fl_dropout(F.relu(self.zl_to_fl_fc1(zl)))))
        fa = F.relu(self.za_to_fa_fc2(self.za_to_fa_dropout(F.relu(self.za_to_fa_fc1(za)))))
        fv = F.relu(self.zv_to_fv_fc2(self.zv_to_fv_dropout(F.relu(self.zv_to_fv_fc1(zv)))))

        fyfl = torch.cat([fy, fl], dim=1)
        fyfa = torch.cat([fy, fa], dim=1)
        fyfv = torch.cat([fy, fv], dim=1)

        dec_len = t
        x_l_hat = self.decoder_l.forward(fyfl, dec_len)
        x_a_hat = self.decoder_a.forward(fyfa, dec_len)
        x_v_hat = self.decoder_v.forward(fyfv, dec_len)
        y_hat = self.fy_to_y_fc2(self.fy_to_y_dropout(F.relu(self.fy_to_y_fc1(fy))))
        decoded = [x_l_hat, x_a_hat, x_v_hat, y_hat]

        return decoded, mmd_loss, missing_loss


if __name__ == '__main__':
    # [{'batchsize': 32, 'num_epochs': 200, 'zv_to_fv_dropout': 0.7,
    # 'memsize': 64, 'fy_size': 16, 'fa_size': 8, 'lr': 0.01,
    # 'zl_to_fl_dropout': 0.2, 'momentum': 0.9, 'fv_size': 8,
    # 'zy_size': 32, 'input_dims': [300, 5, 20], 'zl_size': 32,
    # 'fy_to_y_dropout': 0.0, 'za_size': 8, 'h_dims': [88, 64, 48],
    # 'za_to_fa_dropout': 0.2, 'lda_xa': 0.01, 'fl_size': 88,
    # 'windowsize': 2, 'lda_xl': 1.0, 'zy_to_fy_dropout': 0.0,
    # 'lda_mmd': 1.0, 'zv_size': 80, 'lda_xv': 0.5},
    # {'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5},
    # {'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5},
    # {'shapes': 64, 'drop': 0.5}]
    config = dict()
    config["input_dims"] = [300, 5, 20]
    hl = 88  # random.choice([32,64,88,128,156,256])
    ha = 64  # random.choice([8,16,32,48,64,80])
    hv = 48  # random.choice([8,16,32,48,64,80])
    config["h_dims"] = [hl, ha, hv]
    config['zy_size'] = 32  # random.choice([8,16,32,48,64,80])
    config['zl_size'] = 32  # random.choice([32,64,88,128,156,256])
    config['za_size'] = 8  # random.choice([8,16,32,48,64,80])
    config['zv_size'] = 80  # random.choice([8,16,32,48,64,80])
    config['fy_size'] = 16  # random.choice([8,16,32,48,64,80])
    config['fl_size'] = 88  # random.choice([32,64,88,128,156,256])
    config['fa_size'] = 8  # random.choice([8,16,32,48,64,80])
    config['fv_size'] = 8  # random.choice([8,16,32,48,64,80])
    config["memsize"] = 64  # random.choice([64,128,256,300,400])
    config['zy_to_fy_dropout'] = 0.0  # random.choice([0.0,0.2,0.5,0.7])
    config['zl_to_fl_dropout'] = 0.2  # random.choice([0.0,0.2,0.5,0.7])
    config['za_to_fa_dropout'] = 0.2  # random.choice([0.0,0.2,0.5,0.7])
    config['zv_to_fv_dropout'] = 0.7  # random.choice([0.0,0.2,0.5,0.7])
    config['fy_to_y_dropout'] = 0.0  # random.choice([0.0,0.2,0.5,0.7])

    config['lda_mmd'] = 1.0  # random.choice([0.01,0.1,0.5,1.0])
    config['lda_xl'] = 1.0  # random.choice([0.01,0.1,0.5,1.0])
    config['lda_xa'] = 0.01  # random.choice([0.01,0.1,0.5,1.0])
    config['lda_xv'] = 0.5  # random.choice([0.01,0.1,0.5,1.0])

    config['missing'] = 0
    config["windowsize"] = 2
    config["batchsize"] = 32  # random.choice([32,64,128])
    config["num_epochs"] = 30
    config["lr"] = 0.01  # random.choice([0.001,0.002,0.005,0.008,0.01,0.02])
    config["momentum"] = 0.9  # random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
    NN1Config = dict()
    NN1Config["shapes"] = 128  # random.choice([32,64,128,256])
    NN1Config["drop"] = 0.5  # random.choice([0.0,0.2,0.5,0.7])
    NN2Config = dict()
    NN2Config["shapes"] = 128  # random.choice([32,64,128,256])
    NN2Config["drop"] = 0.5  # random.choice([0.0,0.2,0.5,0.7])
    gamma1Config = dict()
    gamma1Config["shapes"] = 128  # random.choice([32,64,128,256])
    gamma1Config["drop"] = 0.5  # random.choice([0.0,0.2,0.5,0.7])
    gamma2Config = dict()
    gamma2Config["shapes"] = 128  # random.choice([32,64,128,256])
    gamma2Config["drop"] = 0.5  # random.choice([0.0,0.2,0.5,0.7])
    outConfig = dict()
    outConfig["shapes"] = 64  # random.choice([32,64,128,256])
    outConfig["drop"] = 0.5  # random.choice([0.0,0.2,0.5,0.7])
    configs = [config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig]

    model = MFM(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig).cuda()

    t = torch.randn(50, 128, 300).cuda()
    v = torch.randn(50, 128, 20).cuda()
    a = torch.randn(50, 128, 5).cuda()

    x = torch.cat((t, a, v), dim=-1)

    output = model(x)

    print(output[0][-1].shape)  # (batch, 1)
