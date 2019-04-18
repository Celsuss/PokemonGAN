#%%
import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import random
import Utils
import time
import math
import sys
import os

import tensorflow.keras.layers as layers

print('Tensorflow ' + tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

####################
# Create variables #
####################

version = 'newPokemon'
newPoke_path = './' + version
data_path = './data'

checkpoint_path = './checkpoints'
checkpoint_epoch_path = checkpoint_path + '/checkpoint.epoch'

checkpoint_path_old = './checkpoint'
checkpoint_prefix_old = os.path.join(checkpoint_path_old, 'checkpoint.ckpgt')
checkpoint_epoch_path_old = checkpoint_prefix_old + '.epoch'

SAVE_ITERATIONS = 1

DEBUG = False
N_DEBUG_PLOT_IMAGES = 3

noise_dim = 100
num_samples_to_generate = 8*8

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000

generatorFilters = [512, 256, 128, 64]
discriminatorFilters = [64, 128, 256, 512]

generatorLearningRate = 1e-4
discriminatorLearningRate = 1e-4

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
    current_dir = os.getcwd()
    data_root = os.path.join(current_dir, data_path)
    data_root = pathlib.Path(data_root)
    print('Data root directory: {}'.format(data_root))

    all_image_paths = list(data_root.glob('*.*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    print("Number of images: {}".format(image_count))

    return all_image_paths

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

    ds = images_ds.shuffle(buffer_size=len(paths)).batch(BATCH_SIZE)
    n_batch = len(paths)/BATCH_SIZE
    print('ds len: {}, n_batch: {}'.format(len(paths), n_batch))
    
    print('Dataset: {}'.format(ds))

    return ds, n_batch

########################
# Create the Generator #
########################

def createGenerator():
    model = tf.keras.Sequential()
    # model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    # model.add(layers.Dense(7*7*256*3, use_bias=False, input_shape=(noise_dim,)))
    # 4*4*512=8192
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Reshape((7, 7, 256)))
    # model.add(layers.Reshape((7*3, 7*3, 256*3)))
    model.add(layers.Reshape((4, 4, 512)))
    # assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
    # assert model.output_shape == (None, 4, 4, 512) # Note: None is the batch size
    

    # model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 7, 7, 128)
    # assert model.output_shape == (None, 4, 4, 128)  
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    # assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    ## New ones
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    ##

    model.add(layers.Conv2DTranspose(CHANNEL, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

############################
# Create the Discriminator #
############################

def createDiscriminator():
    model = tf.keras.Sequential()
    # model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
    #                                  input_shape=[28, 28, 1]))
    # 128*128*3=49152
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                     input_shape=[HEIGHT, WIDTH, CHANNEL]))
    # 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
      
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    ## New ones
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    ##
       
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

    Utils.generate_and_save_images(images, name, newPoke_path)
    # Utils.saveImages(images, [8,8], newPoke_path)

    if (epoch+1) % SAVE_ITERATIONS == 0:
        # ckpt.step.assign_add(1)
        save_path = ckptManager.save()
        print("Saved checkpoint {}".format(save_path))
        # print("Saved checkpoint for step {}: {}".format(int(ckptManager.step), save_path))

        # checkpoint.save(file_prefix=checkpoint_prefix)
        # checkpoint.save()
        # Save which epoch
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
        

    # if os.path.isfile(checkpoint_epoch_path_old):
    #     with open(checkpoint_epoch_path_old, "rb") as f:
    #         checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path_old))
    #         start_epoch = int(f.read())
    #         print("Restoring previous checkpoint at epoch {}".format(start_epoch+1))
    #         return start_epoch
    # else:
    #     print("No checkpoints found, start training at epoch 0")
    #     return 0

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

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator, discriminator=discriminator)
    ckptManager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)


    # checkpoint_path = './checkpoints'
    # checkpoint_epoch_path = checkpoint_path + '/checkpoint.epoch'

    # checkpoint_path_old = './checkpoint'
    # checkpoint_prefix_old = os.path.join(checkpoint_path_old, 'checkpoint.ckpgt')
    # checkpoint_epoch_path_old = checkpoint_prefix_old + '.epoch'


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
    if len(sys.argv) == 2:
        if sys.argv[1] == '-d':
            global DEBUG
            DEBUG = True
            print("Debug active")

if __name__ == '__main__':
    checkArgs()
    print("PokemonGanV2")
    train()