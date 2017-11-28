import random
import argparse
import numpy as np
from subprocess import call


def get_random_position():
  x = random.uniform(0, 90)
  y = random.uniform(0, 175)
  z = random.uniform(0, 500)
  yaw = random.uniform(0, 360)
  pitch = random.uniform(-89, 89)
  return x, y, z, yaw, pitch


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--list', type=str, help='File with list of camera positions to render')
  #parser.add_argument('--num', type=int, default=10, help='Number of images to collect')
  parser.add_argument('--samples-train', type=int, default=2, help='Samples per pixel for training images')
  parser.add_argument('--samples-gt', type=int, default=20000, help='Samples per pixel for ground truth images')
  args = parser.parse_args()

  data = np.loadtxt(args.list)
  for i in range(data.shape[0]):
    # randomize camera
    x, y, z, yaw, pitch = data[i] #get_random_position()

    # training data
    output_path = 'data/' + str(i) + '_train'
    call(['./pathtrace', '--nobitmap', '-s', str(args.samples_train), '-x', str(x), '-y', str(y), '-z', str(z), '--camera-yaw', str(yaw), '--camera-pitch', str(pitch), '-o', output_path])
    # ground truth
    output_path = 'data/' + str(i) + '_gt'
    call(['./pathtrace', '--nobitmap', '-s', str(args.samples_gt), '-x', str(x), '-y', str(y), '-z', str(z), '--camera-yaw', str(yaw), '--camera-pitch', str(pitch), '-o', output_path])



if __name__ == '__main__':
  main()