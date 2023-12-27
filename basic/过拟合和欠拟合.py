import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

print(tf.__version__)
logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
# tf.data.experimental.CsvDataset 类可用于直接从 Gzip 文件读取 CSV 记录，而无需中间的解压步骤
ds = tf.data.experimental.CsvDataset(gz, [float(), ] * (FEATURES + 1), compression_type="GZIP")


# CSV 读取器类会为每条记录返回一个标量列表。下面的函数会将此标量列表重新打包为 (feature_vector, label) 对。
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label


packed_ds = ds.batch(10000).map(pack_row).unbatch()
for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)


