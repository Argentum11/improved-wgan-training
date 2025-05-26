import tflib.plot
import tflib.cifar10
import tflib.save_images
import tflib.ops.deconv2d
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.linear
import tflib as lib
from tflib.inception_score import get_inception_score_from_generator
import tensorflow as tf
import numpy as np
import time
import os
import sys
sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch_fidelity.datasets")


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = ''
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'wgan-gp'  # Valid options are dcgan, wgan, or wgan-gp
DIM = 128  # This overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 200000  # How many generator iterations to train for
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (3*32*32)

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)


def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random.normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm(
            'Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D(
        'Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm(
            'Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear(
        'Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])


def calculate_discriminator_loss(real_data_int):
    real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
    fake_data = Generator(BATCH_SIZE)

    disc_real = Discriminator(real_data)
    disc_fake = Discriminator(fake_data)
    if MODE == 'wgan':
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    elif MODE == 'wgan-gp':
        # Standard WGAN loss
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random.uniform(
            shape=[BATCH_SIZE, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)

        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            disc_interpolates = Discriminator(interpolates)

        gradients = tape.gradient(disc_interpolates, interpolates)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += LAMBDA * gradient_penalty
    elif MODE == 'dcgan':
        disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_fake, tf.zeros_like(disc_fake)))
        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            disc_real, tf.ones_like(disc_real)))
        disc_cost /= 2.
    return disc_cost


# Set up optimizers
if MODE == 'wgan':
    gen_optimizer = tf.optimizers.RMSprop(learning_rate=5e-5)
    disc_optimizer = tf.optimizers.RMSprop(learning_rate=5e-5)
elif MODE == 'wgan-gp':
    gen_optimizer = tf.optimizers.Adam(
        learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.optimizers.Adam(
        learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
elif MODE == 'dcgan':
    gen_optimizer = tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    disc_optimizer = tf.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)


def train_generator_step():
    with tf.GradientTape() as tape:
        fake_data = Generator(BATCH_SIZE)
        disc_fake = Discriminator(fake_data)
        if MODE == 'wgan' or MODE == 'wgan-gp':
            gen_cost = -tf.reduce_mean(disc_fake)
        elif MODE == 'dcgan':
            gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                disc_fake, tf.ones_like(disc_fake)))

    gen_params = lib.params_with_name('Generator')
    gen_gradients = tape.gradient(gen_cost, gen_params)
    gen_optimizer.apply_gradients(zip(gen_gradients, gen_params))
    return gen_cost


def train_discriminator_step(real_data_int):
    real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)

    with tf.GradientTape() as tape:
        fake_data = Generator(BATCH_SIZE)
        disc_real = Discriminator(real_data)
        disc_fake = Discriminator(fake_data)

        if MODE == 'wgan':
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        elif MODE == 'wgan-gp':
            # Standard WGAN loss
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            # Gradient penalty
            alpha = tf.random.uniform(
                shape=[BATCH_SIZE, 1],
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha * differences)

            with tf.GradientTape(persistent=True) as gp_tape:
                # Watch interpolates for gradient penalty computation
                gp_tape.watch(interpolates)
                disc_interpolates = Discriminator(interpolates)

            gradients = gp_tape.gradient(disc_interpolates, interpolates)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_cost += LAMBDA * gradient_penalty
        elif MODE == 'dcgan':
            disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                disc_fake, tf.zeros_like(disc_fake)))
            disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                disc_real, tf.ones_like(disc_real)))
            disc_cost /= 2.

    disc_params = lib.params_with_name('Discriminator')
    disc_gradients = tape.gradient(disc_cost, disc_params)
    disc_optimizer.apply_gradients(zip(disc_gradients, disc_params))
    return disc_cost

def clip_disc_weights():
    """Clip discriminator weights to [-0.01, 0.01] for WGAN training"""
    disc_params = lib.params_with_name('Discriminator')
    for var in disc_params:
        var.assign(tf.clip_by_value(var, -0.01, 0.01))

# For generating samples
fixed_noise_128 = tf.constant(
    np.random.normal(size=(128, 128)).astype('float32'))


def generate_image(frame, true_dist):
    samples = Generator(128, noise=fixed_noise_128)
    samples = ((samples+1.)*(255./2)).numpy().astype('int32')
    lib.save_images.save_images(samples.reshape(
        (128, 3, 32, 32)), 'samples_{}.jpg'.format(frame))


# Dataset iterators
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR)


def inf_train_gen():
    while True:
        for images, _ in train_gen():
            yield images


# Train loop
gen = inf_train_gen()
for iteration in range(ITERS):
    start_time = time.time()
    # Train generator
    if iteration > 0:
        train_generator_step()
    # Train critic
    if MODE == 'dcgan':
        disc_iters = 1
    else:
        disc_iters = CRITIC_ITERS
    for i in range(disc_iters):
        _data = next(gen)
        _disc_cost = train_discriminator_step(_data)
        if MODE == 'wgan':
            clip_disc_weights()

    lib.plot.plot('train disc cost', _disc_cost)
    lib.plot.plot('time', time.time() - start_time)

    # Calculate inception score every 1K iters
    if iteration % 1000 == 999:
        inception_score = get_inception_score_from_generator(Generator)
        lib.plot.plot('inception score', inception_score)

    # Calculate dev loss and generate samples every 100 iters
    if iteration % 100 == 99:
        dev_disc_costs = []
        for images, _ in dev_gen():
            _dev_disc_cost = calculate_discriminator_loss(images)
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
        generate_image(iteration, _data)

    # Save logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()

    lib.plot.tick()
