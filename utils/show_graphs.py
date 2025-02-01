import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("dark_background")

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

time_window = 50
x_values = np.arange(time_window)
values = [[] for _ in range(6)]

lines = []

colors = ["red", "green", "blue", "brown", "pink", "yellow"]
labels = ["Voltage", "Acceleration (Ax)", "Acceleration (Ay)", "Hall Effect Sensor", "Current", "Signal Strength"]

for i, ax in enumerate(axes.flat):
    line, = ax.plot([], [], color=colors[i], lw=2)
    ax.set_ylim(-10, 10)
    ax.set_xlim(0, time_window - 1)
    ax.set_xticks([])
    ax.set_title(labels[i], color="white")
    ax.tick_params(axis="y", colors="white")
    lines.append(line)

def update(frame):
    for i in range(6):
        if len(values[i]) < time_window:
            values[i].append(np.random.randn())
        else:
            values[i] = values[i][1:] + [values[i][-1] + np.random.randn()]

        lines[i].set_data(range(len(values[i])), values[i])

    return lines

ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)

plt.tight_layout()
plt.show()
