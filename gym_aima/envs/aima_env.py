import sys
from contextlib import closing

import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils
from six import StringIO, b

LEFT, DOWN, RIGHT, UP = range(4)

class AIMAEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, map_name="3x4", noise=0.2, living_rew=0.0, sink=False, render_mode=None):
        self.render_mode = render_mode

        desc = ["FFFG", "FWFH", "SFFF"]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = desc.shape

        self.nA = 4
        self.nS = self.nrow * self.ncol
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        isd = np.array(desc == b"S").astype("float64").ravel()
        self.isd = isd / isd.sum()

        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_s(row, col): return row * self.ncol + col
        def inc(row, col, a):
            if a == LEFT:  col = max(col - 1, 0)
            if a == DOWN:  row = min(row + 1, self.nrow - 1)
            if a == RIGHT: col = min(col + 1, self.ncol - 1)
            if a == UP:    row = max(row - 1, 0)
            return row, col

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                for a in range(self.nA):
                    li = P[s][a]
                    letter = desc[row, col]
                    if sink:
                        if letter in b"W":
                            li.append((1.0, s, 0.0, True))
                        elif letter in b"G":
                            li.append((1.0, 5, 1.0, True))
                        elif letter in b"H":
                            li.append((1.0, 5, -1.0, True))
                        else:
                            for bdir in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, bdir)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                newstate = s if newletter in b"W" else newstate
                                prob = np.round(1.0 - noise if a == bdir else noise / 2.0, 2)
                                li.append((prob, newstate, living_rew, False))
                    else:
                        if letter in b"GHW":
                            li.append((1.0, s, 0.0, True))
                        else:
                            for bdir in [(a - 1) % 4, a, (a + 1) % 4]:
                                newrow, newcol = inc(row, col, bdir)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                newstate = s if newletter in b"W" else newstate
                                terminated = bytes(newletter) in b"GH"
                                rew = living_rew + (1.0 if newletter == b"G" else -1.0 if newletter == b"H" else 0.0)
                                prob = np.round(1.0 - noise if a == bdir else noise / 2.0, 2)
                                li.append((prob, newstate, rew, terminated))

        self.P = P
        self.s = None
        self.lastaction = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = int(self.np_random.choice(self.nS, p=self.isd))
        self.lastaction = None
        return self.s, {}

    def step(self, action):
        transitions = self.P[self.s][int(action)]
        probs = [t[0] for t in transitions]
        i = int(self.np_random.choice(len(transitions), p=probs))
        prob, s, r, terminated = transitions[i]
        self.s = int(s)
        self.lastaction = int(action)
        return self.s, float(r), bool(terminated), False, {"prob": prob}

    def render(self):
        if self.render_mode is None:
            return
        outfile = StringIO() if self.render_mode == "ansi" else sys.stdout
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in self.desc.tolist()]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left','Down','Right','Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")
        if self.render_mode == "ansi":
            with closing(outfile):
                return outfile.getvalue()
