import models
import torch
import torch.nn.functional as F

GAMMA = 0.99
TAU = 1.0

class SacActor:
    def __init__(self, state_dim, num_actions, image_dim, lr=0.001, entropy_coeff=0.2, device="cpu", model_path=None):
        self.device = device

        self.model = models.SAC(state_dim, image_dim, num_actions).to(self.device)
        self.target_model = models.SAC(state_dim, image_dim, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print(f"loaded model from {model_path}")
        self.num_actions = num_actions
        self.episode = 0

        self.entropy_coeff = entropy_coeff

    def set_episode(self, episode):
        self.episode = episode

    def optimize_model(self, batch, weights):
        screenshot_batch, state_batch, action_batch, reward_batch, next_screenshot_batch, next_state_batch, done_batch = batch
        action_batch = action_batch.view(-1, 1)

        actions_logits, state_Q1, state_Q2 = self.model(state_batch, screenshot_batch)
        state_action_values1 = state_Q1.gather(1, action_batch).squeeze(1)
        state_action_values2 = state_Q2.gather(1, action_batch).squeeze(1)
        state_action_values =(state_action_values1 + state_action_values2) / 2.0
        #print("state ", state_batch)
        #print("next state ", next_state_batch)

        with torch.no_grad():
            next_actions_logits, next_state_Q1, next_state_Q2 = self.target_model(next_state_batch, next_screenshot_batch)
            sample_actions = F.softmax(next_actions_logits, dim=1).multinomial(1).detach()
            sampled_log_probs = next_actions_logits.gather(1, sample_actions).detach().squeeze(1)

            q1 = next_state_Q1.gather(1, sample_actions).detach().squeeze(1)
            q2 = next_state_Q2.gather(1, sample_actions).detach().squeeze(1)
            next_state_action_values = torch.min(q1, q2) * (1 - done_batch)
            next_state_action_values = next_state_action_values - self.entropy_coeff * sampled_log_probs

            expected_state_action_values = (next_state_action_values * GAMMA) + reward_batch.squeeze(1)

        critic_loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none') * torch.tensor(weights).to(self.device)
        #print("state action values", state_action_values)
        #print("expected state action values", expected_state_action_values)
       
        highest_value_actions = ((state_Q1+state_Q2)/2.0).argmax(1).detach()
        actor_loss = F.cross_entropy(actions_logits, highest_value_actions, reduction='none')

        #print("actor loss", actor_loss.mean().item())
        #print("critic loss", critic_loss.mean().item())

        retloss = critic_loss + actor_loss
        loss = torch.mean(retloss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        target_net_state_dict = self.target_model.state_dict()
        current_net_state_dict = self.model.state_dict()
        for k in target_net_state_dict.keys():
            target_net_state_dict[k] = TAU * current_net_state_dict[k] + (1 - TAU) * target_net_state_dict[k]
        self.target_model.load_state_dict(target_net_state_dict)

        #print("ret shape", retloss.shape)

        return retloss.detach()

    def save_model(self, path):
        print(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)

    def select_action(self, state, screenshot, t):
        with torch.no_grad():
            self.model.eval()
            screenshot = screenshot.to(self.device)
            state = torch.tensor(state).to(self.device).unsqueeze(0)
            actions_logits, state_Q1, state_Q2 = self.model(state, screenshot)
            state_Q = (state_Q1 + state_Q2) / 2
            actions_probs = F.softmax(actions_logits, dim=1)
            action = actions_probs.multinomial(1).detach()
            self.model.train()
            return action.view(1, 1).cpu().numpy()[0]

    def update_done(self):
        pass
