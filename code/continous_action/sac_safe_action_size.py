import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy,QPrior,ActorPrior,VPrior


class SAC_Safe(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.n_ensemble = args.n_ensemble
        self.lcb_constant = args.lcb
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(self.device)

        self.critic = QPrior(n_ensemble=self.n_ensemble, num_inputs=num_inputs, num_actions=action_space.shape[0],
                             hidden_dim=args.hidden_size).to(self.device)
        self.critic_target = QPrior(n_ensemble=self.n_ensemble, num_inputs=num_inputs,
                                    num_actions=action_space.shape[0], hidden_dim=args.hidden_size).to(self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.safe_critic = QPrior(n_ensemble=self.n_ensemble, num_inputs=num_inputs, num_actions=action_space.shape[0],
                                  hidden_dim=args.hidden_size).to(self.device)
        self.safe_policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
            self.device)

        self.safe_value_network = VPrior(n_ensemble=self.n_ensemble, num_inputs=num_inputs, hidden_dim=args.hidden_size).to(
            self.device)


        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            '''
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)'''
            self.policy = ActorPrior(n_ensemble=self.n_ensemble, num_inputs=num_inputs,
                                     num_actions=action_space.shape[0], hidden_dim=args.hidden_size, action_space=None)
            self.policy_target = ActorPrior(n_ensemble=self.n_ensemble, num_inputs=num_inputs,
                                            num_actions=action_space.shape[0], hidden_dim=args.hidden_size,
                                            action_space=None)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self.load_checkpoint_safe(ckpt_path=args.safe_path)

    def select_action(self, state, evaluate=False, begin=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        value = self.safe_value_network(state)
        value = torch.cat(value, 0)
        value_mean = torch.mean(value, axis=0)
        value_std = torch.std(value, axis=0)
        LCB_safe = value_mean -self.lcb_constant * value_std

        safe_action, _, _ = self.safe_policy(state)

        LCB = None
        action = None
        action_size = 0
        for i in range(5):
            a, _, _ = self.policy(state)
            _,_, q_std, q = self.critic(state, a)

            q = torch.cat(q, 0)
            q_mean = torch.mean(q, axis=0)
            q_std = torch.std(q, axis=0)

            LCB_temp = q_mean - self.lcb_constant * q_std
            if(LCB_temp > LCB_safe):
                action_size = action_size + 1
            if(LCB == None or LCB_temp> LCB):
                LCB = LCB_temp
                action = a


        if evaluate is False:

            if (LCB < LCB_safe):
                return safe_action.detach().cpu().numpy()[0], LCB, LCB_safe, (LCB_safe - LCB), q_mean, value_mean,action_size
            else:
                return action.detach().cpu().numpy()[0], LCB, LCB_safe, (LCB_safe - LCB), q_mean, value_mean,action_size

        else:
            if (begin == True):
                return safe_action.detach().cpu().numpy()[0], LCB, LCB_safe, (LCB_safe - LCB), q_mean, value_mean,action_size
            else:
                return action.detach().cpu().numpy()[0], LCB, LCB_safe, (LCB_safe - LCB), q_mean, value_mean,action_size

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        next_state_action, next_state_log_pi, _ = self.policy(next_state_batch)
        qf1_next_targets, qf2_next_targets, _, _ = self.critic_target(next_state_batch, next_state_action)
        qf1, qf2, uncertainity, _ = self.critic(state_batch, action_batch)
        _, _, safe_uncertainity, _ = self.safe_critic(state_batch, action_batch)

        loss = 0
        for k in range(self.n_ensemble):
            with torch.no_grad():
                min_qf_next_target = torch.min(qf1_next_targets[k],
                                               qf2_next_targets[k]) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1[k],
                                  next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2[k],
                                  next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss
            loss += qf_loss

        # loss =loss / self.n_ensemble
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy(state_batch)
        qf1_pi, qf2_pi, _, _ = self.critic(state_batch, pi)

        policy_loss = 0
        for k in range(self.n_ensemble):
            min_qf_pi = torch.min(qf1_pi[k], qf2_pi[k])
            policy_loss += ((self.alpha * log_pi) - min_qf_pi).mean()
            # policy_loss.append(((self.alpha * log_pi) - min_qf_pi).mean()) # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        policy_loss = policy_loss / self.n_ensemble
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), uncertainity, safe_uncertainity

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                   'critic_state_dict': self.critic.state_dict(),
                   'critic_target_state_dict': self.critic_target.state_dict(),
                   'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                   'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint_safe(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.safe_critic.load_state_dict(checkpoint['critic_state_dict'])
            self.safe_policy.load_state_dict(checkpoint['policy_state_dict'])
            self.safe_value_network.load_state_dict(checkpoint['value_state_dict'])









