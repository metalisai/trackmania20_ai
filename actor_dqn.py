import models
import torch
import random

HIDDEN_DIM = 256
GAMMA = 0.995
TAU = 0.0001

class DqnActor:
    def __init__(self, state_dim, num_actions, image_dim, frame_stack=2, lr=0.0001, device="cpu", model_path=None, dueling=True):

        if dueling:
            policy_net = models.DuelingDQN(state_dim=state_dim, image_dim=image_dim, frame_stack=frame_stack, num_actions=num_actions, num_hidden=HIDDEN_DIM).to(device)
        else:
            policy_net = models.DQN(state_dim=state_dim, image_dim=image_dim, num_actions=num_actions, num_hidden=HIDDEN_DIM).to(device)
        if model_path is not None:
            state_dict = torch.load(model_path)
            policy_net.load_state_dict(state_dict)
            print(f"loaded model from {model_path}")

        if dueling:
            target_net = models.DuelingDQN(state_dim=state_dim, image_dim=image_dim, frame_stack=frame_stack, num_actions=num_actions, num_hidden=HIDDEN_DIM).to(device)
        else:
            target_net = models.DQN(state_dim=state_dim, image_dim=image_dim, num_actions=num_actions, num_hidden=HIDDEN_DIM).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        if dueling:
            active_net = models.DuelingDQN(state_dim=state_dim, image_dim=image_dim, frame_stack=frame_stack, num_actions=num_actions, num_hidden=HIDDEN_DIM).to(device)
        else:
            active_net = models.DQN(state_dim=state_dim, image_dim=image_dim, num_actions=num_actions, num_hidden=HIDDEN_DIM).to(device)
        active_net.load_state_dict(policy_net.state_dict())

        self.device = device

        self.target_net = target_net
        self.policy_net = policy_net
        self.active_net = active_net
        self.num_actions = num_actions

        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, amsgrad=True)

        self.episode = 0
        self.randomness = 0.8

    def set_epsilon(self, epsilon):
        self.randomness = epsilon

    def set_episode(self, ep):
        self.episode = ep
        if ep > 50:
            self.randomness = 0.5
        if ep > 100:
            self.randomness = 0.4
        if ep > 150:
            self.randomness = 0.3
        if ep > 200:
            self.randomness = 0.2
        if ep > 250:
            self.randomness = 0.1

    def optimize_model(self, batch, weights, double_dqn=True, cql=False):
        device = self.device

        if isinstance(batch, list) or isinstance(batch, tuple):
            screenshot_batch, state_batch, action_batch, reward_batch, next_screenshot_batch, next_state_batch, done_batch = batch
            #print(batch)
        else:
            state_batch = batch["state"].to(device)
            screenshot_batch = batch["screenshot"].to(device)
            action_batch = batch["action"].to(device).view(-1, 1)
            next_state_batch = batch["next_state"].to(device)
            next_screenshot_batch = batch["next_screenshot"].to(device)
            reward_batch = batch["reward"].to(device)
            done_batch = batch["done"].to(device)

        action_batch = action_batch.view(-1, 1)

        #state_batch, screenshot_batch, action_batch, next_state_batch, next_screenshot_batch, reward_batch, done_batch = batch

        # compute expected rewards for taken actions
        action_Q = self.policy_net(state_batch, screenshot_batch)
        state_action_values = action_Q.gather(1, action_batch).squeeze(1)

        if cql:
            logsumexp = torch.logsumexp(action_Q, dim=1)
            cql_loss = logsumexp - state_action_values

        if not double_dqn: # seems to work better
            # compute next state values (ignores indices from max())
            target_Q = self.target_net(next_state_batch, next_screenshot_batch).max(1)[0]
            target_Q_masked = target_Q * (1 - done_batch)
            expected_state_action_values = (target_Q_masked * GAMMA) + reward_batch.squeeze(1)
        else:
            # double dqn
            next_state_actions = self.policy_net(next_state_batch, next_screenshot_batch).max(1)[1] # a'
            target_Q = self.target_net(next_state_batch, next_screenshot_batch).gather(1, next_state_actions.unsqueeze(1)).detach().squeeze(1) # Q(s', a')
            target_Q_masked = target_Q * (1 - done_batch)
            expected_state_action_values = (target_Q_masked * GAMMA) + reward_batch.squeeze(1)
        #print(f"expected {torch.mean(target_Q_masked)} actual {torch.mean(state_action_values)}")

        bm_loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values, reduction="none") * torch.tensor(weights).to(self.device)
            
        if cql:
            retloss = 0.5*bm_loss + cql_loss
            loss = torch.mean(retloss)
        else:
            retloss = bm_loss
            loss = torch.mean(retloss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # update target network
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for k in target_net_state_dict.keys():
            target_net_state_dict[k] = TAU * policy_net_state_dict[k] + (1 - TAU) * target_net_state_dict[k]
        self.target_net.load_state_dict(target_net_state_dict)

        #print(f"loss {loss.item()}")
        return retloss.detach()

    def update_done(self):
        self.active_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        print(f"Saving model to {path}")
        torch.save(self.policy_net.state_dict(), path)

    def select_action(self, state, screenshots, t):
        global biased_action_ep, biased_action
        if (self.episode % 2 == 0 and random.random() < self.randomness) or random.random() < 0.05:
            # multinomial distribution
            probs = torch.ones(self.num_actions)
            # forward 2x more often
            probs[3] = 4.0
            ret = torch.multinomial(probs, 1).view(1, 1)
            # to array
            ret = ret.cpu().numpy()
            return ret[0]
        else:
            #print(f"policy {episode}")
            with torch.no_grad():
                device = self.device
                state = torch.tensor(state).unsqueeze(0).to(device)
                scat = torch.cat(screenshots, dim=1).to(device)
                #print(f"scat {scat.shape}")
                #screenshot = screenshot.to(device)
                self.active_net.eval()
                ret = self.active_net(state, scat).max(1)[1].view(1, 1)
                # to array  
                ret = ret.cpu().numpy()
                return ret[0]

