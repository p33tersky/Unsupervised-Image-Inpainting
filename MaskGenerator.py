import numpy as np
import matplotlib.pyplot as plt
import os
import random
import string

mask = np.zeros((16, 16), dtype=np.uint8)
count_marked = 0  

def on_move(event):
    global count_marked
    if event.xdata is not None and event.ydata is not None and event.button == 1:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < 16 and 0 <= y < 16 and mask[y, x] == 0:
            mask[y, x] = 1
            count_marked += 1  
            ax.imshow(mask, cmap="gray")
            plt.draw()
            if count_marked >= 64:
                plt.close() 

fig, ax = plt.subplots()
ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.show()

def random_string_returner(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

npy_path = os.path.join("masksnpy", f"mask_{random_string_returner()}.npy")
png_path = os.path.join("maskspng", f"mask_{random_string_returner()}.png")
np.save(npy_path, mask)
plt.imsave(png_path, mask, cmap="gray")

print(f"Maks saved as '{npy_path}' / '{png_path}'")
