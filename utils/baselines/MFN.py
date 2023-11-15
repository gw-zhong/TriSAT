import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models_transformer import BertTextEncoder


class MFN(nn.Module):
    def __init__(self, config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig):
        super(MFN, self).__init__()

        self.text_model = BertTextEncoder()  # BERT version

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

    def forward(self, x_l, x_a, x_v):
        # x_l = x[:, :, :self.d_l]
        # x_a = x[:, :, self.d_l:self.d_l + self.d_a]
        # x_v = x[:, :, self.d_l + self.d_a:]
        x_l = self.text_model(x_l).transpose(0, 1)
        x = torch.cat((x_l, x_a, x_v), dim=-1)
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
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return output


if __name__ == '__main__':
    # mae 0.993 [{'input_dims': [300, 5, 20], 'batchsize': 128, 'memsize': 128,
    # 'windowsize': 2, 'lr': 0.01, 'num_epochs': 100, 'h_dims': [88, 48, 16], 'momentum': 0.9},
    # {'shapes': 128, 'drop': 0.0}, {'shapes': 64, 'drop': 0.2},
    # {'shapes': 256, 'drop': 0.0}, {'shapes': 64, 'drop': 0.2},
    # {'shapes': 64, 'drop': 0.5}]

    # acc 77.0 [{'input_dims': [300, 5, 20], 'batchsize': 128, 'memsize': 400,
    # 'windowsize': 2, 'lr': 0.005, 'num_epochs': 100, 'h_dims': [64, 8, 80], 'momentum': 0.9},
    # {'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.2},
    # {'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5},
    # {'shapes': 256, 'drop': 0.5}]

    config = dict()
    config["input_dims"] = [300, 5, 20]
    hl = 64
    ha = 8
    hv = 80
    config["h_dims"] = [hl, ha, hv]
    config["memsize"] = 400
    config["windowsize"] = 2
    config["batchsize"] = 128
    config["num_epochs"] = 50
    config["lr"] = 0.005
    config["momentum"] = 0.9
    NN1Config = dict()
    NN1Config["shapes"] = 128
    NN1Config["drop"] = 0.5
    NN2Config = dict()
    NN2Config["shapes"] = 128
    NN2Config["drop"] = 0.2
    gamma1Config = dict()
    gamma1Config["shapes"] = 128
    gamma1Config["drop"] = 0.5
    gamma2Config = dict()
    gamma2Config["shapes"] = 128
    gamma2Config["drop"] = 0.5
    outConfig = dict()
    outConfig["shapes"] = 256
    outConfig["drop"] = 0.5
    configs = [config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig]

    model = MFN(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig).cuda()

    t = torch.randn(50, 128, 300).cuda()
    v = torch.randn(50, 128, 20).cuda()
    a = torch.randn(50, 128, 5).cuda()

    x = torch.cat((t, a, v), dim=-1)

    output = model(x)

    print(output.shape)  # (batch, 1)
