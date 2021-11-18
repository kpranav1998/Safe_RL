import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6



# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)


        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1,x2




class QPrior(nn.Module):
    def __init__(self,  n_ensemble, num_inputs, hidden_dim,num_actions, prior_scale=3):
        super(QPrior, self).__init__()
        self.net = nn.ModuleList([QNetwork(num_inputs, num_actions, hidden_dim) for k in range(n_ensemble)])
        self.prior = nn.ModuleList([QNetwork(num_inputs, num_actions, hidden_dim) for k in range(n_ensemble)])
        self.n_ensemble = n_ensemble
        # used when scaling core net
        self.prior_scale = prior_scale

    def forward(self, state, actions):
        a1 = []
        a2 = []
        min_vals = []
        for k in range(self.n_ensemble):
            n1,n2 = self.net[k](state,actions[k])
            p1,p2 = self.prior[k](state,actions[k])
            x = n1 + self.prior_scale * p1.detach()
            y = n2 + self.prior_scale * p2.detach()
            a1.append(x)
            a2.append(y)
            min_vals.append(torch.min(x,y))

            #ans.append(self.net[k](state,action) + self.prior_scale * self.prior[k](state,action).detach())

        std = torch.cat(min_vals, 0)
        std = torch.std(std, axis=0)
        return a1,a2,std,min_vals


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class ActorPrior(nn.Module):
    def __init__(self,n_ensemble, num_inputs, num_actions, hidden_dim,  prior_scale=3,action_space=None):
        super(ActorPrior, self).__init__()

        # used when scaling core net
        self.net = nn.ModuleList([GaussianPolicy(num_inputs, num_actions, hidden_dim, action_space=action_space) for k in range(n_ensemble)])
        #self.prior = nn.ModuleList([GaussianPolicy(num_inputs, num_actions, hidden_dim, action_space=action_space) for k in range(n_ensemble)])

        self.prior_scale = prior_scale

    def forward(self, state):
        actions = []
        log_probs = []
        means = []
        for k in range(len(self.net)):
            action, log_prob, mean = self.net[k](state)
            #action_prior, log_prob_prior, mean_prior = self.prior[k](state)
            action = action #+ self.prior_scale *action_prior.detach()
            log_prob = log_prob #+ self.prior_scale *log_prob_prior.detach()
            mean = mean #+ self.prior_scale *mean_prior.detach()

            actions.append(action)
            log_probs.append(log_prob)
            means.append(mean)
        return actions,log_probs,means,torch.mean(torch.cat(actions, axis=0), 0) , torch.mean(torch.cat(log_probs, axis=0), 0) ,torch.mean(torch.cat(means, axis=0), 0)