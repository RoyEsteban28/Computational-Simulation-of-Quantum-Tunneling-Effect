import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from utils import barrier, gaussian, tunneling_probability


class WavePacket:
    k0 = np.pi / 20
    PA = 0
    PR = 1
    FU = 2
    counter = 0

    def __init__(
        self,
        spatial_points=800,
        dx=1.0e0,
        m=1.0e0,
        hbar=1.0e0,
        barrier_height=1.0e-2,
        barrier_width=30,
        sigma=40.0,
    ):
        self.N = spatial_points
        self.dx = dx
        self.m = m
        self.hbar = hbar
        self.X = self.dx * np.linspace(
            -spatial_points // 2, spatial_points // 2, spatial_points
        )
        self.V0 = barrier_height
        self.THCK = barrier_width
        self.sigma = sigma

        self.x0 = -(round(spatial_points / 2) - 5 * sigma)
        self.E = (hbar ** 2 / 2.0 / m) * (self.k0 ** 2 + 0.5 / sigma ** 2)

        self.V = barrier(self.X, barrier_height, barrier_width)
        self.vmax = self.V.max()
        self.dt = hbar / (2 * hbar ** 2 / (m * dx ** 2) + self.vmax)
        self.c1 = hbar * self.dt / (m * dx ** 2)
        self.c2 = 2 * self.dt / hbar
        self.cv2 = self.c2 * self.V

        self.psi_r = np.zeros((3, self.N))
        self.psi_i = np.zeros((3, self.N))
        self.psi_p = np.zeros(
            self.N,
        )

        self.IDX1 = range(1, self.N - 1)
        self.IDX2 = range(2, self.N)
        self.IDX3 = range(0, self.N - 2)
        self.start()

    def start(self):
        self.psi_r = np.zeros((3, self.N))
        self.psi_i = np.zeros((3, self.N))
        self.psi_p = np.zeros(
            self.N,
        )
        xn = range(-1 - self.N // 2, self.N // 2)
        x = self.X[xn] / self.dx

        gg = gaussian(x, self.x0, self.sigma)
        cx = np.cos(self.k0 * x)
        sx = np.sin(self.k0 * x)
        self.psi_r[self.PR, xn] = cx * gg
        self.psi_i[self.PR, xn] = sx * gg
        self.psi_r[self.PA, xn] = cx * gg
        self.psi_i[self.PA, xn] = sx * gg

        psi_p = self.psi_r[self.PR] ** 2 + self.psi_i[self.PR] ** 2

        self.P = self.dx * psi_p.sum()
        nrm = np.sqrt(self.P)
        self.psi_r /= nrm
        self.psi_i /= nrm
        self.psi_p /= self.P

    def evolve(self, i):
        self.counter += 1
        for _ in range(i):

            psi_r_pr = self.psi_r[self.PR]
            psi_i_pr = self.psi_i[self.PR]

            self.psi_i[self.FU, self.IDX1] = self.psi_i[
                self.PA, self.IDX1
            ] + self.c1 * (
                psi_r_pr[self.IDX2] - 2 *
                psi_r_pr[self.IDX1] + psi_r_pr[self.IDX3]
            )
            self.psi_i[self.FU] -= self.cv2 * self.psi_r[self.PR]

            self.psi_r[self.FU, self.IDX1] = self.psi_r[
                self.PA, self.IDX1
            ] - self.c1 * (
                psi_i_pr[self.IDX2] - 2 *
                psi_i_pr[self.IDX1] + psi_i_pr[self.IDX3]
            )
            self.psi_r[self.FU] += self.cv2 * self.psi_i[self.PR]

            self.psi_r[self.PA] = psi_r_pr
            self.psi_r[self.PR] = self.psi_r[self.FU]
            self.psi_i[self.PA] = psi_i_pr
            self.psi_i[self.PR] = self.psi_i[self.FU]

        psi_p = self.psi_r[self.PR] ** 2 + self.psi_i[self.PR] ** 2
        return psi_p


class Animator:
    PR = 1
    counter = 0

    def __init__(self, wave_packet):
        self.wave_packet = wave_packet
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(self.wave_packet.X.min(), self.wave_packet.X.max())
        ymax = 1.5 * (self.wave_packet.psi_r[self.PR]).max()
        self.ax.set_ylim(-ymax, ymax)
        (self.line_real,) = self.ax.plot([], [], c="y",  alpha=0.5)
        (self.line_imaginary,) = self.ax.plot(
            [], [], c="m",  alpha=0.5)
        (self.line_prob,) = self.ax.plot(
            [], [], c="b", label=r"$|\psi(x)|^2$", linewidth=2
        )
        (self.barrier_line,) = self.ax.plot([], [], c="r", label=r"$V(x)$")

        self.title = self.ax.set_title("")
        self.ax.legend(prop=dict(size=12))
        self.ax.set_xlabel("$x$")
        self.ax.set_ylabel(r"$|\psi(x)|$")

        efac = ymax / 2.0 / self.wave_packet.vmax
        v_plot = self.wave_packet.V * efac
        self.barrier_line.set_data(self.wave_packet.X, v_plot)

    def init(self):
        self.line_real.set_data([], [])
        self.line_imaginary.set_data([], [])
        self.line_prob.set_data([], [])
        self.title.set_text("")
        return (self.line_real, self.line_imaginary, self.line_prob)

    def update(self, data):

        self.line_real.set_data(
            self.wave_packet.X, self.wave_packet.psi_r[self.PR])
        self.line_imaginary.set_data(
            self.wave_packet.X, self.wave_packet.psi_i[self.PR]
        )
        self.line_prob.set_data(self.wave_packet.X, 6 * data)
        self.counter += 1
        return self.line_real, self.line_imaginary, self.line_prob

    def time_step(self):
        while True:
            if self.counter == 250:
                self.wave_packet.start()
                self.counter = 0
            yield self.wave_packet.evolve(30)

    def animate(self, save=False):
        self.ani = animation.FuncAnimation(
            self.fig, self.update, self.time_step, interval=20, blit=False, repeat=False, save_count=250
        )
        if save:
            with open("index.html", "w") as f:
                print(self.ani.to_jshtml(), file=f)


if __name__ == "__main__":
    wave_packet = WavePacket()
    animator = Animator(wave_packet)
    animator.animate(save=True)
    plt.show()
