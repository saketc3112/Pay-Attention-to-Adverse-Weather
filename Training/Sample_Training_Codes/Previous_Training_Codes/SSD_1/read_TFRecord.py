import tensorflow as tf

raw_dataset = tf.data.TFRecordDataset("/data/datasets/saket/SeeingThroughFogDataset/train_clear_day/train_clear_day_000545.swedentfrecord")

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
