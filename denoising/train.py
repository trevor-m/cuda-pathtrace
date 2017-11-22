import tensorflow as tf
import numpy as np
import argparse
import skimage.io
from load_data import load_exr_data

NUM_FEATURE_CHANNELS = 14

def save_image(path, image):
  """Takes an image with range 0.0 to 1.0 and shape (1, w, h, 3) and saves to disk""" 
  if len(image.shape) == 4:
    image = image.reshape((image.shape[1], image.shape[2], image.shape[3]))
  image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
  skimage.io.imsave(path, image)

def build_model():
  """Set up the model here"""
  input_features = tf.placeholder(tf.float32, [None, None, None, NUM_FEATURE_CHANNELS], name='input_features')
  gt_color = tf.placeholder(tf.float32, [None, None, None, 3], name='gt_color')
  
  x = tf.layers.conv2d(input_features, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
  x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
  x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
  denoised_color = tf.layers.conv2d(x, filters=3, kernel_size=(3,3), padding='same')

  loss = tf.reduce_sum(tf.abs(denoised_color - gt_color))
  train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

  return denoised_color, loss, train_step

def test_model(iteration, sess, output, loss, data_in, data_gt):
  """Compute the loss and save out the output image"""
  test_loss = sess.run(loss, feed_dict={'input_features:0': data_in, 'gt_color:0': data_gt})
  test_output = sess.run(output, feed_dict={'input_features:0': data_in, 'gt_color:0': data_gt})
  save_image(str(iteration)+"_out.png", test_output)
  save_image(str(iteration)+"_gt.png", data_gt)
  return test_loss

def main():
  # create model
  model_output, model_loss, model_train = build_model()
  
  # load data
  data = load_exr_data('../4.exr')
  train_in = np.concatenate(data, axis=2)
  train_in = train_in.reshape((1, train_in.shape[0], train_in.shape[1], train_in.shape[2]))
  train_gt = load_exr_data('../20k.exr')[0]
  train_gt = train_gt.reshape((1, train_gt.shape[0], train_gt.shape[1], train_gt.shape[2]))

  # start session
  sess = tf.Session()
  sess.run(tf.local_variables_initializer())
  sess.run(tf.global_variables_initializer())

  for iteration in range(99999999):
    # test every 1000 iterations
    if iteration % 100 == 0:
      loss = test_model(iteration, sess, model_output, model_loss, train_in, train_gt)
      print('[%d] Loss: %.7f' % (iteration, loss))
    
    # train
    train_loss = sess.run(model_train, feed_dict={'input_features:0': train_in, 'gt_color:0': train_gt})

if __name__ == '__main__':
  main()