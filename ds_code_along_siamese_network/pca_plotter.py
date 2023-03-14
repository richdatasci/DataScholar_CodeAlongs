# PCA Plotter for Refreshing Visual for PCA 
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

class PCAPlotter(tf.keras.callbacks.Callback):
  def __init__(self, plt, embedding_model, x_test, y_test):
    super(PCAPlotter, self).__init__()
    self.embedding_mode = embedding_model
    self.x_test = x_test
    self.y_test = y_test
    self.fig = plt.figure(figsize=(12,6))
    self.ax1 = plt.subplot(1, 2, 1)
    self.ax2 = plt.subplot(1, 2, 2)
    plt.ion()

    self.losses = []

  def plot(self, epoch=None, plot_loss=False):
    x_test_embeddings = self.embedding_model.predict(self.x_test)
    pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
    self.ax1.clear()
    self.ax1.scatter(pca_out[:, 0], c=self.y_test, cmap='seismic')
    if plot_loss:
      self.ax2.clear()
      self.ax2.plot(range(epoch), self.losses)
      self.ax2.set_xlabel('epochs')
      self.ax2.set_ylabel('loss')
    self.fig.canvas.draw()

  def on_train_begin(self, logs=None):
    self.losses = []
    self.fig.show()
    self.fig.canvas.draw()
    self.plot()
  
  def on_epoch_end(self, epoch, logs=None):
    self.losses.append(logs.get('loss')
    self.plot(epoch+1, plot_loss=True)