import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

# Load the LiDAR BIN file using numpy
data = np.fromfile('E:/Study/jason/sfa3d/newdata/New folder/00000.bin', dtype=np.float32)
data = data.reshape(-1, 4)

df = pd.DataFrame(data, columns=['x', 'y', 'z', 'intensity'])

# Convert the data to a PyntCloud object
cloud = PyntCloud(df, info={"name": "lidar"})

# Visualize the PyntCloud object using the "open3d" backend
cloud.plot(point_size=0.02, backend="open3d")
