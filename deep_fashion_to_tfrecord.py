import tensorflow as tf
import pandas as pd
from PIL import Image

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('categories', '', 'Define the level of categories. broad or fine.')
flags.DEFINE_string('evaluation_status', '', 'train, val or test')
FLAGS = flags.FLAGS

LABEL_DICT = {1: "top", 2: "bottom", 3: "long"}


def create_tf_example(example, path_root):

  width, height = Image.open(path_root + example["image_name"]).size

  filename = example["image_name"] # Filename of the image. Empty if image is not from file
  filename = filename.encode()

  with tf.gfile.GFile(path_root + example['image_name'], 'rb') as fid:
        encoded_image_data = fid.read()

  image_format = 'jpeg'.encode() # b'jpeg' or b'png'

  xmins = [example['x_1']/width] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [example['x_2']/width] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [example['y_1']/height] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [example['y_2']/height] # List of normalized bottom y coordinates in bounding box
             # (1 per box)

  if FLAGS.categories == 'broad':
      classes_text = [LABEL_DICT[example['category_type']].encode()] # List of string class name of bounding box (1 per box)
      classes = [example['category_type']] # List of integer class id of bounding box (1 per box)
  elif FLAGS.categories == 'fine':
      classes_text = [example['category_name'].encode()] # List of string class name of bounding box (1 per box)
      classes = [example['category_label']] # List of integer class id of bounding box (1 per box)
  else:
      raise(ValueError("Incorrect value for flag categories. Must be 'broad' or 'fine'."))

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # Annotation file paths
  bbox_file = "Category and Attribute Prediction Benchmark/Anno/list_bbox.txt"
  cat_cloth_file = "Category and Attribute Prediction Benchmark/Anno/list_category_cloth.txt"
  cat_img_file = "Category and Attribute Prediction Benchmark/Anno/list_category_img.txt"
  stage_file = "Category and Attribute Prediction Benchmark/Eval/list_eval_partition.txt"
  # Read annotation files
  bbox_df = pd.read_csv(bbox_file, sep='\s+', skiprows=1)
  cat_cloth_df = pd.read_csv(cat_cloth_file, sep='\s+', skiprows=1)
  cat_img_df = pd.read_csv(cat_img_file, sep='\s+', skiprows=1)
  stage_df = pd.read_csv(stage_file, sep='\s+', skiprows=1)
  # Merge dfs
  cat_cloth_df["category_label"] = cat_cloth_df.index + 1
  cat_df = cat_img_df.merge(cat_cloth_df, how='left', on='category_label')
  examples_df = cat_df.merge(bbox_df, how='left', on='image_name')
  examples_df = examples_df.merge(stage_df, how='left', on='image_name')
  # to remove -->
  examples_df = examples_df[examples_df["evaluation_status"] == FLAGS.evaluation_status]
  examples_df = examples_df.head(1000)

  for irow, example in examples_df.iterrows():
    tf_example = create_tf_example(example, path_root="Category and Attribute Prediction Benchmark/Img/")
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
