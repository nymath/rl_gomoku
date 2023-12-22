import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = nn.BatchNorm2d(in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x
    
    
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            drop_out=0.1,
            use_attention=False,
    ):
        super().__init__()
        self.drop_out = drop_out
        self.activation = F.relu
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Sequential(
            nn.Dropout(p=drop_out),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.norm_2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_connection = nn.Identity()

        if use_attention:
            self.attention = AttentionBlock(out_channels)
        else:
            self.attention = nn.Identity()
        
    def forward(self, x):
        out = self.norm_1(self.conv_1(x))
        out = self.activation(out)
        out = self.norm_2(self.conv_2(out))
        out = self.activation(out + self.residual_connection(x))
        out = self.attention(out)
        return out


class AlphaZeroValueNet(nn.Module):
    def __init__(self, board_size, in_channels, dropout=0.1, use_attention=False):
        super().__init__()
        self.board_size = board_size

        # shared network
        self.shared_network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64, dropout, use_attention),
            ResidualBlock(64, 128, dropout, use_attention),        
        )

        # policy branch
        self.policy_branch = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*self.board_size*self.board_size, board_size*board_size),
            nn.LogSoftmax(dim=1),  # here we return the normalized logits
        )

        # state-value branch
        self.value_branch = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # bound the value to (-1, 1)
        )

    def forward(self, x):
        state = self.shared_network(x)
        act_logits = self.policy_branch(state)
        act_value = self.value_branch(state)
        return act_logits, act_value


def mcts_train(net: nn.Module, optimizer: torch.optim.Optimizer, data, device="cuda"):
    states = torch.FloatTensor(np.array(data["states"])).to(device)          # (batch_size, 7, 15, 15)
    mcts_probs = torch.FloatTensor(np.array(data["mcts_probs"])).to(device)  # (batch_size, 225)
    winners = torch.FloatTensor(np.array(data["winners"])).to(device)        # (batch_size,)

    act_logits, state_value = net(states)
    value_loss = F.mse_loss(state_value.squeeze(), winners.squeeze())
    policy_loss = - torch.sum(mcts_probs*F.softmax(act_logits), 1)   # the cross entropy
    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def actor_critic_train(net: nn.Module, optimizer: torch.optim.Optimizer, transition_dict, device="cuda"):
    # on policy
    pass


def ppo_train(net: nn.Module, optimizer: torch.optim.Optimizer, replay_buffer, device="cuda"):
    pass


class NaiveNet(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val