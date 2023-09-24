from torch import nn
import torch

# ConvNet for DQN

# input_shape: (batch_size, 3, 224, 224)
# output_shape: (batch_size, 28*28)
class ConvNet(nn.Module):
    def __init__(self, grayscale=False):
        super(ConvNet, self).__init__()
        
        if grayscale:
            in_channels = 1
        else:
            in_channels = 3
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        #self.bn3 = nn.BatchNorm2d(64)
        #self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.bn4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        #x = nn.functional.relu(x)
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = nn.functional.relu(x)
        #x = self.conv4(x)
        #x = self.bn4(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        return x

class DQN(nn.Module):
    def __init__(self, state_dim, image_dim, num_actions, num_hidden, grayscale=False):
        super(DQN, self).__init__()
        self.num_actions = num_actions

        self.conv = ConvNet(grayscale=grayscale)
        in_channels = 1 if grayscale else 3
        final_conv_size = self.conv(torch.zeros(1, in_channels, image_dim, image_dim)).shape[1]
        self.fc1 = nn.Linear(state_dim + final_conv_size, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        #self.bn2 = nn.BatchNorm1d(num_hidden)
        #self.fc3 = nn.Linear(num_hidden, num_hidden)
        #self.bn3 = nn.BatchNorm1d(num_hidden)
        #self.fc4 = nn.Linear(num_hidden, num_actions)


    def forward(self, state, screen):
        x = self.conv(screen)
        x = torch.cat([x, state], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        #x = nn.functional.relu(x)
        #x = self.fc3(x)
        #x = self.bn3(x)
        #x = nn.functional.relu(x)
        #x = self.fc4(x)
        return x

class SAC(nn.Module):
    def __init__(self, state_dim, image_dim, num_actions):
        super(SAC, self).__init__()
        self.num_actions = num_actions

        self.conv = ConvNet(grayscale=True)
        final_conv_size = self.conv(torch.zeros(1, 1, image_dim, image_dim)).shape[1]

        num_hidden = 256

        self.fc = nn.Linear(state_dim + final_conv_size, 512)
    
        # actor branch (outputs <num_actions> probabilities)
        self.actor_fc1 = nn.Linear(512, num_hidden)
        self.actor_bn1 = nn.BatchNorm1d(num_hidden)
        self.actor_fc2 = nn.Linear(num_hidden, num_hidden)
        self.actor_bn2 = nn.BatchNorm1d(num_hidden)
        self.actor_fc3 = nn.Linear(num_hidden, num_actions)

        self.critic1_fc1 = nn.Linear(512, num_hidden)
        self.critic1_bn1 = nn.BatchNorm1d(num_hidden)
        self.critic1_fc2 = nn.Linear(num_hidden, num_hidden)
        self.critic1_bn2 = nn.BatchNorm1d(num_hidden)
        self.critic1_fc3 = nn.Linear(num_hidden, num_actions)

        self.critic2_fc1 = nn.Linear(512, num_hidden)
        self.critic2_bn1 = nn.BatchNorm1d(num_hidden)
        self.critic2_fc2 = nn.Linear(num_hidden, num_hidden)
        self.critic2_bn2 = nn.BatchNorm1d(num_hidden)
        self.critic2_fc3 = nn.Linear(num_hidden, num_actions)

    def forward(self, state, ss):
        x = self.conv(ss)
        x = torch.cat([x, state], dim=1)
        x = self.fc(x)

        x_actor = self.actor_fc1(x)
        x_actor = self.actor_bn1(x_actor)
        x_actor = nn.functional.relu(x_actor)
        x_actor = self.actor_fc2(x_actor)
        x_actor = self.actor_bn2(x_actor)
        x_actor = nn.functional.relu(x_actor)
        x_actor = self.actor_fc3(x_actor)

        x_critic1 = self.critic1_fc1(x)
        x_critic1 = self.critic1_bn1(x_critic1)
        x_critic1 = nn.functional.relu(x_critic1)
        x_critic1 = self.critic1_fc2(x_critic1)
        x_critic1 = self.critic1_bn2(x_critic1)
        x_critic1 = nn.functional.relu(x_critic1)
        x_critic1 = self.critic1_fc3(x_critic1)

        x_critic2 = self.critic2_fc1(x)
        x_critic2 = self.critic2_bn1(x_critic2)
        x_critic2 = nn.functional.relu(x_critic2)
        x_critic2 = self.critic2_fc2(x_critic2)
        x_critic2 = self.critic2_bn2(x_critic2)
        x_critic2 = nn.functional.relu(x_critic2)
        x_critic2 = self.critic2_fc3(x_critic2)

        return x_actor, x_critic1, x_critic2
