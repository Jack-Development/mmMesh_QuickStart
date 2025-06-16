import torch
import torch.nn as nn


class AnchorPointNet(nn.Module):
    def __init__(self):
        super(AnchorPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24 + 4 + 3, out_channels=32, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))  # (Batch, feature, frame_point_number)

        x = x.transpose(1, 2)

        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(x * attn_weights, dim=1)
        return attn_vec, attn_weights


class AnchorVoxelNet(nn.Module):
    def __init__(self):
        super(AnchorVoxelNet, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=64, out_channels=96, kernel_size=(3, 3, 3), padding=(0, 0, 0)
        )
        self.cb1 = nn.BatchNorm3d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=96, out_channels=128, kernel_size=(5, 1, 1))
        self.cb2 = nn.BatchNorm3d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 1, 1))
        self.cb3 = nn.BatchNorm3d(64)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.permute(0, 4, 1, 2, 3)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.view(batch_size, 64)
        return x


class AnchorRNN(nn.Module):
    def __init__(self):
        super(AnchorRNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )

    def forward(self, x, h0, c0):
        a_vec, (hn, cn) = self.rnn(x, (h0, c0))
        return a_vec, hn, cn


class BasePointNet(nn.Module):
    def __init__(self):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)
        x = torch.cat((in_mat[:, :, :4], x), -1)

        return x


class GlobalPointNet(nn.Module):
    def __init__(self):
        super(GlobalPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24 + 4, out_channels=32, kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1, 2)

        attn_weights = self.softmax(self.attn(x))
        attn_vec = torch.sum(x * attn_weights, dim=1)
        return attn_vec, attn_weights


class GlobalRNN(nn.Module):
    def __init__(self):
        super(GlobalRNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(64, 16)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x, h0, c0):
        g_vec, (hn, cn) = self.rnn(x, (h0, c0))
        g_loc = self.fc1(g_vec)
        g_loc = self.faf1(g_loc)
        g_loc = self.fc2(g_loc)
        return g_vec, g_loc, hn, cn
