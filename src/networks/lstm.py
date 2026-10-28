"""LSTM building blocks: cell, actor-critic network."""

from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class LSTMCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        h, c = carry
        combined = jnp.concatenate([x, h], axis=-1)
        gates = nn.Dense(4 * self.hidden_size, kernel_init=nn.initializers.orthogonal())(combined)
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = nn.sigmoid(i)
        f = nn.sigmoid(f)
        g = jnp.tanh(g)
        o = nn.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)
        return (new_h, new_c), new_h


class ActorCriticLSTM(nn.Module):
    action_size: int
    lstm_hidden_size: int = 512
    actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    init_noise_std: float = 1.0

    def setup(self):
        self.actor_lstm = LSTMCell(self.lstm_hidden_size)
        self.critic_lstm = LSTMCell(self.lstm_hidden_size)
        self.actor_mlp = [
            nn.Dense(dim, kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2.0)))
            for dim in self.actor_hidden_dims
        ]
        self.actor_out = nn.Dense(self.action_size, kernel_init=nn.initializers.orthogonal(scale=0.01))
        self.critic_mlp = [
            nn.Dense(dim, kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2.0)))
            for dim in self.critic_hidden_dims
        ]
        self.critic_out = nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=0.01))
        self.std = self.param("std", nn.initializers.constant(self.init_noise_std), (self.action_size,))

    def __call__(self, actor_obs, critic_obs, actor_hidden, critic_hidden):
        new_actor_hidden, actor_out = self.actor_lstm(actor_hidden, actor_obs)
        x = actor_out
        for layer in self.actor_mlp:
            x = nn.tanh(layer(x))
        action_mean = self.actor_out(x)
        action_std = jnp.abs(self.std) + 1e-8

        new_critic_hidden, critic_out = self.critic_lstm(critic_hidden, critic_obs)
        x = critic_out
        for layer in self.critic_mlp:
            x = nn.tanh(layer(x))
        value = self.critic_out(x).squeeze(-1)
        return action_mean, action_std, value, new_actor_hidden, new_critic_hidden

    def act(self, actor_obs, actor_hidden):
        new_actor_hidden, actor_out = self.actor_lstm(actor_hidden, actor_obs)
        x = actor_out
        for layer in self.actor_mlp:
            x = nn.tanh(layer(x))
        action_mean = self.actor_out(x)
        action_std = jnp.abs(self.std) + 1e-8
        return action_mean, action_std, new_actor_hidden
