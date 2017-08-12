import model
import tensorflow as tf
from pipeline import input_pipeline

TRAINING_DATASET = 'NGII_Training.tfrecords'
BATCH_SIZE = 16
NUM_EPOCHS = 100

x_batch, y_batch = input_pipeline(TRAINING_DATASET, BATCH_SIZE, NUM_EPOCHS)

deconv = model.Deconv('deconv', x_batch, y_batch)

deconv.train(NUM_EPOCHS)
