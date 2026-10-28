"""Odometer (velocity estimation) LSTM network."""

from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from .lstm import LSTMCell


class OdomLSTM(nn.Module):
    odom_output_size: int = 3
    lstm_hidden_size: int = 256
    hidden_dims: Tuple[int, ...] = (256, 256, 256)

    def setup(self):
        self.odom_lstm = LSTMCell(self.lstm_hidden_size)
        self.odom_mlp = [
            nn.Dense(dim, kernel_init=nn.initializers.orthogonal(scale=np.sqrt(2.0)))
            for dim in self.hidden_dims
        ]
        self.odom_out = nn.Dense(
            self.odom_output_size,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        )

    def __call__(self, odom_obs, hidden):
        new_hidden, lstm_out = self.odom_lstm(hidden, odom_obs)
        x = lstm_out
        for layer in self.odom_mlp:
            x = nn.tanh(layer(x))
        odom_data = self.odom_out(x)
        return odom_data, new_hidden
