import tensorflow as tf
import numpy as np
import argparse
import skimage.io
import datetime
import time
import os
import random
from load_data import load_exr_data, get_patches

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
  
  with tf.variable_scope("denoiser"):
    x = tf.layers.conv2d(input_features, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    #x = tf.layers.conv2d(x, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
    denoised_color = tf.layers.conv2d(x, filters=3, kernel_size=(3,3), padding='same')

  loss = tf.reduce_sum(tf.abs(denoised_color - gt_color))
  train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

  return denoised_color, loss, train_step

def test_model(iteration, sess, log_path, output, loss, data_in, data_gt):
  """Compute the loss and save out the output image"""
  test_loss = sess.run(loss, feed_dict={'input_features:0': data_in, 'gt_color:0': data_gt})
  start = time.time()
  test_output = sess.run(output, feed_dict={'input_features:0': data_in, 'gt_color:0': data_gt})
  pred_time = time.time()-start
  save_image(log_path+'/'+str(iteration)+"_out.png", test_output)
  save_image(log_path+'/'+str(iteration)+"_gt.png", data_gt)
  save_image(log_path+'/'+str(iteration)+"_in.png", data_in[:, :, :, 0:3])
  return test_loss, pred_time

def build_output_dir(name=None):
  log_path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  if name:
    log_path += '_' + name
  if not os.path.exists(log_path):
    os.makedirs(log_path)
  return log_path

def get_batch(data, gt, batch_size):
  data, gt = get_patches(data, gt, patch_size=64, num_patches=batch_size)
  idx = np.random.choice(len(data), batch_size, replace=False)
  return np.stack(data[idx], 0), np.stack(gt[idx], 0)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--load', type=str, help='File to load saved weights from')
  parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
  parser.add_argument('--test-freq', type=int, default=1000, help='Will log training status after this many iterations')
  parser.add_argument('--name', type=str, help='Name for the output folder (a timestamp will be used if not specified')
  parser.add_argument('--test-only', action='store_true', help='If set, will only test the model and then exit (no training)')
  args = parser.parse_args()

  # create model
  with tf.device('/device:GPU:1'):
    model_output, model_loss, model_train = build_model()
  
  # load training data
  train_data = load_exr_data('../4.exr', preprocess=True, concat=True)
  train_gt = load_exr_data('../20k.exr', preprocess=True)[0]
  #print(train_data.shape, train_gt.shape)
  #for i in range(train_gt.shape[0]):
  #  save_image('patches/patch'+str(i)+'.png', train_gt[i])
  #return
  # load testing data
  test_in = load_exr_data('../4.exr', preprocess=True, concat=True)
  test_in = test_in.reshape((1, test_in.shape[0], test_in.shape[1], test_in.shape[2]))
  # load color layer only for gt
  test_gt = load_exr_data('../20k.exr', preprocess=True)[0]
  test_gt = test_gt.reshape((1, test_gt.shape[0], test_gt.shape[1], test_gt.shape[2]))
  
  # start session
  sess = tf.Session()
  sess.run(tf.local_variables_initializer())
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.trainable_variables())
  if args.load:
    saver.restore(sess, args.load)

  # create log path
  log_path = build_output_dir(args.name)
  print('Logging results for this session in folder "%s"' % log_path)

  for iteration in range(99999999):
    # test every 1000 iterations
    if iteration % args.test_freq == 0:
      loss, pred_time = test_model(iteration, sess, log_path, model_output, model_loss, test_in, test_gt)
      print('[%d] Loss: %.7f - Prediction Time: %.5fs' % (iteration, loss, pred_time))
      # save weights
      saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)
      if args.test_only:
        return
    # train
    data, gt = get_batch(train_data, train_gt, batch_size=args.batch_size)
    train_loss = sess.run(model_train, feed_dict={'input_features:0': data, 'gt_color:0': gt})

if __name__ == '__main__':
  main()