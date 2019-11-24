#%%
import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import random
import Utils
import time
import math
import glob
import sys
import os

import tensorflow.keras.layers as layers

print('Tensorflow ' + tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

AUTOTUNE = tf.data.experimental.AUTOTUNE

####################
# Create variables #
####################

newPoke_path = './newPokemon'
data_path = 'data/training'

checkpoint_path = './checkpoints'
checkpoint_epoch_path = checkpoint_path + '/checkpoint.epoch'

SAVE_ITERATIONS = 10
SAVE_IMAGE_ITERATION = 10

DEBUG = False
N_DEBUG_PLOT_IMAGES = 3

noise_dim = 100
num_samples_to_generate = 8*8

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 10000

generatorFilters = [512, 256, 128, 64]
discriminatorFilters = [64, 128, 256, 512]

generatorLearningRate = 1e-4
discriminatorLearningRate = 1e-4

image_endings = ['jpg', 'jpeg', 'png']

#%%
#############################
# Functions to print images #
#############################
def plotImage(image):
    plt.imshow(image)
    plt.grid(False)
    plt.show()
    print()

def plotImages(images):
    for n in range(len(images)):
        image = loadAndPreprocessImage(images[n])
        plotImage(image)

def plotImagesDS(images_ds):
    plt.figure(figsize=(8,8))
    for n,image in enumerate(images_ds.take(4)):
        plt.subplot(2,2,n+1)
        plt.imshow(image)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()

#%%
#########################################
# Functions to load and preprocess data #
#########################################
def loadAndPreprocessImage(file_path):
    img_raw = tf.io.read_file(file_path)
    # img_tensor = tf.image.decode_image(img_raw, channels=CHANNEL)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=CHANNEL)
    img_final = tf.image.resize(img_tensor, [HEIGHT, WIDTH])
    img_final /= 255.0
    return img_final

def getImagePaths():
    print('Start loading and process data')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(current_dir, data_path)
    pokemon_files = []
    [pokemon_files.extend(glob.glob(os.path.join(data_root, '*/*.'+i))) for i in image_endings]

    image_count = len(pokemon_files)
    print("Number of images: {}".format(image_count))

    return pokemon_files

def loadDataset():
    paths = getImagePaths()
    # if DEBUG:
    #     plotImages(paths[:N_DEBUG_PLOT_IMAGES])

    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    images_ds = paths_ds.map(loadAndPreprocessImage, num_parallel_calls=AUTOTUNE)
    if DEBUG:
        plotImagesDS(images_ds)

    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    # ds = images_ds.shuffle(buffer_size=len(paths))
    # ds = ds.repeat()
    # ds = ds.batch(BATCH_SIZE)
    # # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    # ds = ds.prefetch(buffer_size=AUTOTUNE)

    data_base = images_ds.shuffle(buffer_size=len(paths)).batch(BATCH_SIZE)
    n_batch = len(paths)/BATCH_SIZE
    print('ds len: {}, n_batch: {}'.format(len(paths), n_batch))
    
    print('Dataset: {}'.format(data_base))

    return data_base, n_batch

########################
# Create the Generator #
########################

def createGenerator():
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 512)))    

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(CHANNEL, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

############################
# Create the Discriminator #
############################

def createDiscriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                     input_shape=[HEIGHT, WIDTH, CHANNEL]))
    
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
      
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
       
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
     
    return model

############################################################
# Restore checkpoint, Save checkpoint and generated images #
############################################################
def saveCheckpointAndImage(ckptManager, generator, seed, epoch):
    # Notice `training` is set to False. 
    # This is so all layers run in inference mode (batchnorm).
    # The same noice is used every time so it's possible to create a gif from the pictures.
    images = generator(seed, training=False)
    name = 'epoch{:04d}.png'.format(epoch+1)

    if (epoch+1) % SAVE_IMAGE_ITERATION == 0:
        Utils.generate_and_save_images(images, name, newPoke_path)
    # Utils.saveImages(images, [8,8], newPoke_path)

    if (epoch+1) % SAVE_ITERATIONS == 0:
        save_path = ckptManager.save()
        print("Saved checkpoint {} epoch {}".format(save_path, epoch+1))
        with open(checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))

def restoreModel(checkpoint, ckptManager):
    checkpoint.restore(ckptManager.latest_checkpoint)
    if ckptManager.latest_checkpoint:
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print('Restored checkpoint from {} at epoch {}'.format(ckptManager.latest_checkpoint, start_epoch))
        return start_epoch
    else:
        print('No checkpoints found, start training at epoch 0')
        return 0

######################
# Train the networks #
######################

def discriminatorLoss(real_output, fake_output, cross_entropy):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

def generatorLoss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def train():
    data, n_batch = loadDataset()

    generator = createGenerator()
    discriminator = createDiscriminator()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(generatorLearningRate)
    discriminator_optimizer = tf.keras.optimizers.Adam(discriminatorLearningRate)

    # Create a Checkpoint and a CheckpointManager
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)
    ckptManager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=2)

    # We will use this seed when we save the images but not when we train.
    seed = tf.random.normal([num_samples_to_generate, noise_dim])

    # Restore the model if there is any checkpoint.
    start_epoch = restoreModel(checkpoint, ckptManager)

    # Strt training
    for epoch in range(start_epoch, EPOCH):
        startTime = time.time()

        batch_count = 0
        for imageBatch in data:
            batch_count += 1
            gen_loss, disc_loss = trainStep(imageBatch, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy)
            print("Epoch {}, Batch {}/{}, Gen loss {}, Disc loss {}".format(epoch+1, batch_count, math.ceil(n_batch), gen_loss, disc_loss))

        display.clear_output()

        # saveCheckpointAndImage(checkpoint, generator, seed, epoch)
        saveCheckpointAndImage(ckptManager, generator, seed, epoch)
        print('Time for epoch {} is {} sec\n'.format(epoch+1, time.time()-startTime))
                

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def trainStep(images, generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)

        fake_output = discriminator(generated_images, training=True)

        gen_loss = generatorLoss(fake_output, cross_entropy)
        disc_loss = discriminatorLoss(real_output, fake_output, cross_entropy)

    gradients_of_generator = genTape.gradient(gen_loss, generator.trainable_variables)
    graidents_of_discriminator = discTape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(graidents_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


##############
# Startpoint #
##############

def checkArgs():
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-d':
            global DEBUG
            DEBUG = True
            print("Debug active")
        elif sys.argv[i] == '-p' or sys.argv[i] == '--path':
            data_path == sys.argv[i+1]
            print('Setting path to: {}'.format(data_path))
        elif sys.argv[i] == '-c' or sys.argv[i] == '--config':
            checkpoint_path = sys.argv[i+1]
        elif sys.argv[i] == '-s' or sys.argv[i] == '--save':
            newPoke_path = sys.argv[i+1]

if __name__ == '__main__':
    checkArgs()
    print("PokemonGanV2")
    train()