config = {  
    'img_size': 64,
    'batch_size': 128,
    'g_lr': 0.0002,
    'g_beta1': 0.5,
    'g_beta2': 0.99,
    'noise_dim': 128,
    'g_initializer': 'orthogonal',
    'g_dim': [1024, 256, 64],
    'g_heads': [4, 4, 4],
    'g_mlp': [512, 512, 512],
    'd_initializer': 'orthogonal',
    'd_enc_dim': [64, 128, 256], # 64x64 [64, 128, 256], 128x128  [32, 64, 128, 256],
    'd_out_dim': [512, 1024],   
    'd_heads': 4,
    'd_mlp': 512,
    'd_lr': 0.0002,
    'd_beta1': 0.5,
    'd_beta2': 0.99,
    'gp_weight': 0.0001, 
    'policy': 'color,translation',
    'fid_batch_size': 50, # inception 
    'gen_batch_size': 50, 
    'loss': 'nsl',
    'ema_decay': 0.999,
    'bcr': False,
    'cr_weight': 0.1,
    'dec_dim': False, # conv decoder 64x64 [32], 128x128 [32, 16], 256x256 [32, 16, 8]
    'n_fid_real': 2500,
    'n_fid_gen': 2500,
    'plot_size': 5.2,
    'test_seed': 77,
}
"""LadaGAN model for Tensorflow.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import tensorflow as tf
from tensorflow.keras import layers


def pixel_upsample(x, H, W):
    B, N, C = x.shape
    assert N == H*W
    x = tf.reshape(x, (-1, H, W, C))
    x = tf.nn.depth_to_space(x, 2, data_format='NHWC')
    B, H, W, C = x.shape
    
    return x, H, W, C


class SMLayerNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, initializer='orthogonal'):
        super(SMLayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.initializer = initializer
        
    def build(self, inputs):
        input_shape, _ = inputs
        self.h = layers.Dense(input_shape[2], use_bias=True, 
                    kernel_initializer=self.initializer
        )
        self.gamma = layers.Dense(input_shape[2], use_bias=True, 
            kernel_initializer=self.initializer, 
        )
        self.beta = layers.Dense(input_shape[2], use_bias=True, 
                        kernel_initializer=self.initializer
        )
        self.ln = layers.LayerNormalization(
            epsilon=self.epsilon, center=False, scale=False
        )

    def call(self, inputs):
        x, z = inputs
        x = self.ln(x)
        h = self.h(z)
        h = tf.nn.relu(h)
        
        scale = self.gamma(h)
        shift = self.beta(h)
        x *= tf.expand_dims(scale, 1)
        x += tf.expand_dims(shift, 1)
        return x


class AdditiveAttention(layers.Layer):
    def __init__(self, model_dim, n_heads, initializer='orthogonal'):
        super(AdditiveAttention, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wk = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wv = layers.Dense(model_dim, kernel_initializer=initializer)
        
        self.q_attn = layers.Dense(n_heads, kernel_initializer=initializer)
        dim_head = model_dim // n_heads

        self.to_out = layers.Dense(model_dim, kernel_initializer=initializer)

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        B = tf.shape(q)[0]
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  
        attn = tf.transpose(self.q_attn(q), [0, 2, 1]) / self.depth ** 0.5
        attn = tf.nn.softmax(attn, axis=-1)  
   
        q = self.split_into_heads(q, B)  
        k = self.split_into_heads(k, B)  
        v = self.split_into_heads(v, B)

        # calculate global vector
        global_q = tf.einsum('b h n, b h n d -> b h d', attn, q) 
        global_q = tf.expand_dims(global_q, 2)
       
        p = global_q * k 
        r = p * v

        r = tf.transpose(r, perm=[0, 2, 1, 3]) 
        original_size_attention = tf.reshape(r, (B, -1, self.model_dim)) 

        output = self.to_out(original_size_attention) 
        return output, attn


class SMLadaformer(layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='orthogonal'):
        super(SMLadaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu', 
                         kernel_initializer=initializer), 
            layers.Dense(model_dim, kernel_initializer=initializer),
        ])
        self.norm1 = SMLayerNormalization(epsilon=eps, initializer=initializer)
        self.norm2 = SMLayerNormalization(epsilon=eps, initializer=initializer)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training):
        inputs, z = x
        x_norm1 = self.norm1([inputs, z])
        
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output, training=training) 
        
        x_norm2 = self.norm2([attn_output, z])
        mlp_output = self.mlp(x_norm2)
        return self.drop2(mlp_output, training=training), attn_maps 
    
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, n_patches, model_dim, initializer='orthogonal'):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = layers.Embedding(
            input_dim=n_patches, output_dim=model_dim, 
            embeddings_initializer=initializer
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return patches + self.position_embedding(positions)
    

class Generator(tf.keras.models.Model):
    def __init__(self, img_size=32, model_dim=[1024, 256, 64], heads=[2, 2, 2], 
                 mlp_dim=[2048, 1024, 512], initializer='orthogonal', dec_dim=False):
        super(Generator, self).__init__()
        self.init = tf.keras.Sequential([
            layers.Dense(8 * 8 * model_dim[0], use_bias=False, 
                                kernel_initializer=initializer),
            layers.Reshape((8 * 8, model_dim[0]))
        ])     
    
        self.pos_emb_8 = PositionalEmbedding(64, model_dim[0], 
                                initializer=initializer)
        self.block_8 = SMLadaformer(model_dim[0], heads[0], 
                                mlp_dim[0], initializer=initializer)
        self.conv_8 = layers.Conv2D(model_dim[1], 3, padding='same', 
                                kernel_initializer=initializer)

        self.pos_emb_16 = PositionalEmbedding(256, model_dim[1], 
                                initializer=initializer)
        self.block_16 = SMLadaformer(model_dim[1], heads[1], 
                                mlp_dim[1], initializer=initializer)
        self.conv_16 = layers.Conv2D(model_dim[2], 3, padding='same', 
                                kernel_initializer=initializer)

        self.pos_emb_32 = PositionalEmbedding(1024, model_dim[2], 
                                initializer=initializer)
        self.block_32 = SMLadaformer(model_dim[2], heads[2], 
                                mlp_dim[2], initializer=initializer)

        self.dec_dim = dec_dim
        if self.dec_dim:
            self.dec = tf.keras.Sequential()
            for _ in self.dec_dim:
                self.dec.add(layers.UpSampling2D(2, interpolation='nearest'))
                self.dec.add(layers.Conv2D(_, 3, padding='same', 
                                    kernel_initializer=initializer)),
                self.dec.add(layers.BatchNormalization())
                self.dec.add(layers.LeakyReLU(0.2))
        else:
            self.patch_size = img_size // 32
        self.ch_conv = layers.Conv2D(3, 3, padding='same', 
                                kernel_initializer=initializer)

    def call(self, z):
        B = z.shape[0]
   
        x = self.init(z)
        x = self.pos_emb_8(x)
        x, attn_8 = self.block_8([x, z])

        x, H, W, C = pixel_upsample(x, 8, 8)
        x = self.conv_8(x)
        x = tf.reshape(x, (B, H * W, -1))

        x = self.pos_emb_16(x)
        x, attn_16 = self.block_16([x, z])

        x, H, W, C = pixel_upsample(x, H, W)
        x = self.conv_16(x)
        x = tf.reshape(x, (B, H * W, -1))
        x = self.pos_emb_32(x)
        x, attn_32 = self.block_32([x, z])

        x = tf.reshape(x, [B, 32, 32, -1])
        if  self.dec_dim:
            x = self.dec(x)
        elif self.patch_size != 1:
            x = tf.nn.depth_to_space(x, self.patch_size, data_format='NHWC')
        return [self.ch_conv(x), [attn_8, attn_16, attn_32]]

    
class downBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, strides=2, 
                 initializer='glorot_uniform'):
        super(downBlock, self).__init__()
        self.main = tf.keras.Sequential([
            layers.Conv2D(filters, kernel_size=kernel_size, 
                padding='same', kernel_initializer=initializer,
                strides=strides, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(filters, kernel_size=3, 
                padding='same', kernel_initializer=initializer,
                strides=1, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

        self.direct = tf.keras.Sequential([
           layers.AveragePooling2D(pool_size=(strides, strides)),
            layers.Conv2D(filters, kernel_size=1, 
                padding='same', kernel_initializer=initializer,
                strides=1, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

    def call(self, x):
        return (self.main(x) + self.direct(x)) / 2


class Ladaformer(tf.keras.layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, 
                 rate=0.0, eps=1e-6, initializer='orthogonal'):
        super(Ladaformer, self).__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = tf.keras.Sequential([
            layers.Dense(
                mlp_dim, activation='gelu', kernel_initializer=initializer
            ), 
            layers.Dense(model_dim, kernel_initializer=initializer),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=eps)
        self.norm2 = layers.LayerNormalization(epsilon=eps)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training):
        x_norm1 = self.norm1(inputs)
        
        attn_output, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_output = inputs + self.drop1(attn_output, training=training) 
        
        x_norm2 = self.norm2(attn_output)
        mlp_output = self.mlp(x_norm2)
        return self.drop2(mlp_output, training=training) + attn_output, attn_maps

    
class Discriminator(tf.keras.models.Model):
    def __init__(self, img_size=32, enc_dim=[64, 128, 256], out_dim=[512, 1024], mlp_dim=512, 
                 heads=2, initializer='orthogonal'):
        super(Discriminator, self).__init__()
        if img_size == 32:
            assert len(enc_dim) == 2, "Incorrect length of enc_dim for img_size 32"
        elif img_size == 64:
            assert len(enc_dim) == 3, "Incorrect length of enc_dim for img_size 64"
        elif img_size == 128:
            assert len(enc_dim) == 4, "Incorrect length of enc_dim for img_size 128"
        elif img_size == 256:
            assert len(enc_dim) == 5, "Incorrect length of enc_dim for img_size 256"
        else:
            raise ValueError(f"img_size = {img_size} not supported")
            
        self.enc_dim = enc_dim
        self.inp_conv = tf.keras.Sequential([
            layers.Conv2D(enc_dim[0], kernel_size=3, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='same'),
            layers.LeakyReLU(0.2),
        ])    
        self.encoder = [downBlock(
            i, kernel_size=3, strides=2, initializer=initializer
        ) for i in enc_dim[1:]]

        self.pos_emb_8 = PositionalEmbedding(256, enc_dim[-1], 
                            initializer=initializer)
        self.block_8 = Ladaformer(enc_dim[-1], heads, 
                            mlp_dim, initializer=initializer)
        
        self.conv_4 = layers.Conv2D(out_dim[0], 3, padding='same', 
                                    kernel_initializer=initializer)
        self.down_4 = tf.keras.Sequential([
            layers.Conv2D(out_dim[1], kernel_size=1, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='valid'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=4, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='valid')
        ])
        '''Logits'''
        self.logits = tf.keras.Sequential([
            layers.Flatten(),
            layers.Activation('linear', dtype='float32')    
        ])
    
    def call(self, img):
        x = self.inp_conv(img)  
        for i in range(len(self.enc_dim[1:])):
            x = self.encoder[i](x)

        B, H, W, C = x.shape
        x = tf.reshape(x, (B, H * W, C))
        x = self.pos_emb_8(x)
        x, maps_16 = self.block_8(x)

        x = tf.reshape(x, (B, H, W, C))
        x = tf.nn.space_to_depth(x, 2, data_format='NHWC') 
        x = self.conv_4(x)

        x = self.down_4(x)
        return [self.logits(x)]
import os
import json
import tensorflow as tf
from tensorflow.keras import layers
from huggingface_hub import hf_hub_download
import json


AUTOTUNE = tf.data.experimental.AUTOTUNE


def deprocess(img):
    return img * 127.5 + 127.5

def train_convert(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = (img - 127.5) / 127.5 
    return img

def create_train_iter_ds(train_dir, batch_size, img_size):
    img_paths = tf.data.Dataset.list_files(str(train_dir))
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)

    img_paths = img_paths.cache().shuffle(BUFFER_SIZE)
    ds = img_paths.map(lambda img: train_convert(img, img_size), 
            num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True, 
            num_parallel_calls=AUTOTUNE)
    print(f'Train dataset size: {BUFFER_SIZE}')
    print(f'Train batches: {tf.data.experimental.cardinality(ds)}')
    ds = ds.repeat().prefetch(AUTOTUNE)
    return iter(ds)

def get_loss(loss):
    if loss == 'nsl':
        def discriminator_loss(real_img, fake_img):
            real_loss = tf.reduce_mean(tf.math.softplus(-real_img))
            fake_loss = tf.reduce_mean(tf.math.softplus(fake_img)) 
            return real_loss + fake_loss

        def generator_loss(fake_img):
            return tf.reduce_mean(tf.math.softplus(-fake_img))

        return generator_loss, discriminator_loss

    elif loss == 'hinge':
        def d_real_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 - logits))

        def d_fake_loss(logits):
            return tf.reduce_mean(tf.nn.relu(1.0 + logits))

        def discriminator_loss(real_img, fake_img):
            real_loss = d_real_loss(real_img)
            fake_loss = d_fake_loss(fake_img)
            return fake_loss + real_loss

        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)

        return generator_loss, discriminator_loss

class Config(object):
    def __init__(self, save_dir, input_dict=None):
        if input_dict != None:
            for key, value in input_dict.items():
                setattr(self, key, value)
        file_path = os.path.join(save_dir, "config.json")

        # Check if the configuration file exists
        if os.path.exists(file_path):
            self.load_config(file_path)
        else:
            self.save_config(file_path, save_dir)

    def save_config(self, file_path, save_dir):
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert input_dict to JSON and save to file
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)
        print(f'New config {file_path} saved')

    def load_config(self, file_path):
        # Load configuration from the existing file
        with open(file_path, "r") as f:
            config_data = json.load(f)

        print(f'Config {file_path} loaded')
        # Update the object's attributes with loaded configuration
        for key, value in config_data.items():
            print(f'{key}: {value}')
            setattr(self, key, value)
        
        
class Loader(object):
    def __init__(self):
        pass
        
    def download(self, ckpt_dir):
        repo_id = 'milmor/LadaGAN'
        if ckpt_dir == 'ffhq_128':
            n_images = 24064000
        elif ckpt_dir == 'bedroom_128':
            n_images = 10624000
        elif ckpt_dir == 'celeba_64':
            n_images = 72192000
        elif ckpt_dir == 'cifar10':
            n_images = 68096000

        hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-{n_images}.data-00000-of-00001",
            local_dir='./'
        )

        hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/best-training-checkpoints/ckpt-{n_images}.index",
            local_dir='./'
        )

        hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/best-training-checkpoints/checkpoint",
            local_dir='./')

        config_file = hf_hub_download(repo_id=repo_id, 
            filename=f"{ckpt_dir}/config.json",
            local_dir='./'
        )

        with open(config_file) as f:
            self.config = json.load(f)
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utils import deprocess
import math


def save_generator_heads(model, epoch, noise, main_dir, 
			resolution_dirs, heads, size=15, n_resolutions=3):
    predictions, maps = model(noise, training=False)
    predictions = np.clip(deprocess(predictions), 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(size, size))

    for i in range(predictions.shape[0]):
        # create subplot and append to ax
        fig.add_subplot(8, 8, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    path = os.path.join(main_dir, f'{epoch:04d}.png')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(path, format='png')
    plt.close()
    
    for r in range(n_resolutions):
        for h in range(heads[r]):
            map_size = int(math.sqrt(maps[r][0][0].shape[0])) # get map high and width 
            fig = plt.figure(figsize=(size, size))
            for i in range(predictions.shape[0]):
                fig.add_subplot(8, 8, i+1)
                map_reshape = tf.reshape(maps[r][i][h], [map_size, map_size])
                plt.imshow(map_reshape)
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            path = os.path.join(resolution_dirs[r], f'ep{epoch:04d}_r{r}_h{str(h)}.png')
            plt.savefig(path)
            plt.close()

def get_map(maps, resolution, head):
    h = head
    r = resolution
    map_size = int(math.sqrt(maps[r][0][0].shape[0])) # get map high and width 
    b = maps[r][:, h].shape[0]
    reshaped_maps = tf.reshape(maps[r][:, h], [b, map_size, map_size, 1])
    return reshaped_maps

def plot_single_head(predictions, maps, h=1, size=1, path=None):
    n = len(predictions)
    maps32 = get_map(maps, 2, h)
    maps16 = get_map(maps, 1, h)
    maps8 = get_map(maps, 0, h)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, n, figsize=(n*size, 4*size))

    # Plot the images in the first row
    for i in range(n):
        axes[0, i].imshow(predictions[i])
        axes[0, i].axis('off')

    # Plot maps32 in the second row
    for i in range(n):
        axes[1, i].imshow(maps32[i, :, :, 0])
        axes[1, i].axis('off')

    # Plot maps16 in the third row
    for i in range(n):
        axes[2, i].imshow(maps16[i, :, :, 0])
        axes[2, i].axis('off')

    # Plot maps8 in the fourth row
    for i in range(n):
        axes[3, i].imshow(maps8[i, :, :, 0])
        axes[3, i].axis('off')

    # Remove the titles
    for ax in axes.ravel():
        ax.set_title("")

    # Display the plot
    plt.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.0)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.show()
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import time
import tensorflow as tf
import json
from tqdm import tqdm
from PIL import Image
from model import Generator, Discriminator
from utils import *
from plot_utils import save_generator_heads
from trainer import LadaGAN
from config import config


def train(file_pattern, eval_dir, model_dir, metrics_inter, 
          fid_inter, total_iter, max_ckpt_to_keep, conf):
    train_ds = create_train_iter_ds(
        file_pattern, conf.batch_size, conf.img_size
    )

    # init model
    noise = tf.random.normal([conf.batch_size, conf.noise_dim])
    generator = Generator(
        img_size=conf.img_size, model_dim=conf.g_dim, 
        heads=conf.g_heads, mlp_dim=conf.g_mlp, dec_dim=conf.dec_dim
    )
    gen_batch = generator(noise)
    generator.summary()
    print('G output shape:', gen_batch[0].shape)
    
    discriminator = Discriminator(
        img_size=conf.img_size, enc_dim=conf.d_enc_dim,   
        out_dim=conf.d_out_dim, heads=conf.d_heads,
        mlp_dim=conf.d_mlp, initializer=conf.d_initializer
    )
    out_disc = discriminator(
        tf.ones([conf.batch_size, conf.img_size, conf.img_size, 3])
    )
    discriminator.summary()
    print('D Output shape:', out_disc[0].shape)
    
    gan = LadaGAN(
        generator=generator, discriminator=discriminator, conf=conf
    )
    
    # define losses
    generator_loss, discriminator_loss = get_loss(conf.loss)

    gan.build(
        g_optimizer=tf.keras.optimizers.Adam(
            learning_rate=conf.g_lr, 
            beta_1=conf.g_beta1, 
            beta_2=conf.g_beta2
        ),
        d_optimizer=tf.keras.optimizers.Adam(
            learning_rate=conf.d_lr, 
            beta_1=conf.d_beta1,
            beta_2=conf.d_beta2
        ),
        g_loss=generator_loss,
        d_loss=discriminator_loss)

    gan.create_ckpt(model_dir, max_ckpt_to_keep, restore_best=False)
    
    # plot seed and plot dir
    num_examples_to_generate = 64 # plot images
    noise_seed = tf.random.normal(
        [num_examples_to_generate, conf.noise_dim], seed=conf.test_seed
    )
    gen_img_dir = os.path.join(model_dir, 'log-gen-img')
    os.makedirs(gen_img_dir, exist_ok=True)
    # additive attention maps plot dirs for 3 stages
    n_resolutions = 3
    resolution_dirs = []
    for resolution in range(n_resolutions):
        path = os.path.join(gen_img_dir, 'res_{}'.format(resolution))
        os.makedirs(path, exist_ok=True)                    
        resolution_dirs.append(path)
    
    # train
    start_iter = int((gan.ckpt.n_images / gan.batch_size) + 1)
    n_images = int(gan.ckpt.n_images)
    start = time.time()
    for idx in range(start_iter, total_iter):
        image_batch = train_ds.get_next()
        gan.train_step(image_batch)

        if idx % metrics_inter == 0:
            print(f'\nTime for metrics_inter is {time.time()-start:.4f} sec')
            n_images = idx * gan.batch_size
            gan.save_metrics(n_images)
            start = time.time()

        if idx % fid_inter == 0:
            n_images = idx * gan.batch_size
            gan.save_ckpt(n_images, conf.n_fid_real, 
                conf.fid_batch_size, eval_dir, conf.img_size
            )
            save_generator_heads(
                gan.ema_generator, n_images, noise_seed, 
                gen_img_dir, resolution_dirs, conf.g_heads, size=conf.plot_size
            )
            start = time.time()

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pattern', type=str)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='model_1')
    parser.add_argument('--metrics_inter', type=int, default=500)
    parser.add_argument('--fid_inter', type=int, default=500)
    parser.add_argument('--total_iter', type=int, default=10000000000)
    parser.add_argument('--max_ckpt_to_keep', type=int, default=2)
    args = parser.parse_args()

    conf = Config(args.model_dir, config)

    train(
        args.file_pattern, args.eval_dir, args.model_dir, 
        args.metrics_inter, args.fid_inter, args.total_iter,
        args.max_ckpt_to_keep, conf
    )


if __name__ == '__main__':
    main()
import numpy as np
from scipy import linalg
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    
def get_activations(dataset, inception, batch_size=20):
    n_batches = tf.data.experimental.cardinality(dataset)
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048), 'float32')

    for i, batch in enumerate(tqdm(dataset)):
        start = i * batch_size
        end = start + batch_size
        pred = inception(batch)
        pred_arr[start:end] = pred # pred.reshape(batch_size, -1)
        
    return pred_arr

def calculate_activation_statistics(images, model, batch_size=20):
    act = get_activations(images, model, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_fid(real, gen, model, batch_size=20):
    if isinstance(real, list):
        m1, s1 = real
    else:   
        m1, s1 = calculate_activation_statistics(real, model, batch_size)
        
    if isinstance(gen, list):
        m2, s2 = gen
    else:  
        m2, s2 = calculate_activation_statistics(gen, model, batch_size)
        
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def test_convert(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    return img

def create_fid_ds(img_dir, batch_size, img_size, n_images, seed=42):
    img_paths = tf.data.Dataset.list_files(str(img_dir), seed=seed).take(n_images)
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)
    ds = img_paths.map(lambda img: test_convert(img, img_size), 
                       num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True, 
            num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    ds_size = tf.data.experimental.cardinality(ds)
    print(f'FID dataset size: {BUFFER_SIZE} FID batches: {ds_size}')  
    return ds

class Inception(tf.keras.models.Model):
    ''' Calculates the activations
    -- inp (inception_v3.preprocess_input): A floating point numpy.array or a tf.Tensor,
            3D or 4D with 3 color channels, with values in the range [0, 255].
    '''
    def __init__(self):
        super(Inception, self).__init__()
        self.res = preprocessing.Resizing(299, 299)
        self.inception = InceptionV3(include_top=False,  pooling='avg')
        self.inception.trainable = False

    @tf.function
    def call(self, inp):
        x = self.res(inp)
        x = inception_v3.preprocess_input(x)       
        x = self.inception(x)
        return x

# Copyright (c) 2020, Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Differentiable Augmentation for Tensorflow.

Reference:
  - [Differentiable Augmentation for Data-Efficient GAN Training](
      https://arxiv.org/abs/2006.10738) (NeurIPS 2020)
"""
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


def DiffAugment(x, policy='', channels_first=False):
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x


def rand_brightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def rand_cutout(x, ratio=0.5):
    #if tf.random.uniform([], minval=0.0, maxval=1.0) < 0.3:
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

"""LadaGAN model for Tensorflow.

Reference:
  - [Efficient generative adversarial networks using linear 
    additive-attention Transformers](https://arxiv.org/abs/2401.09596)
"""
import tensorflow as tf
import os
from tqdm import tqdm
from PIL import Image
import time
from diffaug import DiffAugment
from utils import deprocess
from fid import *


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.mean_squared_error(y_true, y_pred)
    )

def reset_metrics(metrics):
    for _, metric in metrics.items():
        metric.reset_states()

def update_metrics(metrics, **kwargs):
    for metric_name, metric_value in kwargs.items():
        metrics[metric_name].update_state(metric_value)

        
class LadaGAN(object):
    def __init__(self, generator, discriminator, conf):
        super(LadaGAN, self).__init__()
        self.generator = generator
        self.ema_generator = generator
        self.discriminator = discriminator
        self.noise_dim = conf.noise_dim
        self.gp_weight = conf.gp_weight
        self.policy = conf.policy
        self.batch_size = conf.batch_size
        self.ema_decay = conf.ema_decay
        self.ema_generator = tf.keras.models.clone_model(generator)
        self.bcr = conf.bcr
        self.cr_weight = conf.cr_weight
        # init ema
        noise = tf.random.normal([1, conf.noise_dim])
        gen_batch = self.ema_generator(noise)

        # metrics
        self.train_metrics = {}
        self.fid_avg = tf.keras.metrics.Mean()

    def build(self, g_optimizer, d_optimizer, g_loss, d_loss):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss = g_loss
        self.d_loss = d_loss
        self._build_metrics()

    def _build_metrics(self):
        metric_names = [
        'g_loss',
        'd_loss',
        'gp',
        'd_total',
        'real_acc',
        'fake_acc',
        'cr'
        ]
        for metric_name in metric_names:
            self.train_metrics[metric_name] = tf.keras.metrics.Mean()

    def gradient_penalty(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        with tf.GradientTape() as gradient_tape:
            gradient_tape.watch(real_samples)
            logits = self.discriminator(real_samples, training=True)[0]

        r1_grads = gradient_tape.gradient(logits, real_samples)
        r1_grads = tf.reshape(r1_grads, (batch_size, -1))
        r1_penalty = tf.reduce_sum(tf.square(r1_grads), axis=-1)
        r1_penalty = tf.reduce_mean(r1_penalty) * self.gp_weight 
        return logits, r1_penalty

    @tf.function(jit_compile=True)
    def train_step(self, real_images):

        noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
        # train the discriminator
        with tf.GradientTape() as d_tape:
            if self.bcr:
                fake_images = self.generator(noise, training=True)[0]
                fake_logits = self.discriminator(fake_images, training=True)[0]
                real_logits, gp = self.gradient_penalty(real_images) 
                real_augmented_images = DiffAugment(real_images, policy=self.policy)
                fake_augmented_images = DiffAugment(fake_images, policy=self.policy)  
                real_augmented_images = tf.stop_gradient(real_augmented_images)
                fake_augmented_images = tf.stop_gradient(fake_augmented_images)
                
                augmented_images = tf.concat(
                    (real_augmented_images, fake_augmented_images), axis=0)
                augmented_logits = self.discriminator(augmented_images, training=True)[0]
                real_augmented_logits, fake_augmented_logits = tf.split(
                    augmented_logits, num_or_size_splits=2, axis=0)
                consistency_loss = self.cr_weight * (
                    l2_loss(real_logits, real_augmented_logits) +
                    l2_loss(fake_logits, fake_augmented_logits))                
        
                d_loss = self.d_loss(real_logits, fake_logits)
                d_total = d_loss + gp + consistency_loss
                
            else:
                fake_images = self.generator(noise, training=True)[0]
                fake_augmented_images = DiffAugment(fake_images, policy=self.policy) 
                real_augmented_images = DiffAugment(real_images, policy=self.policy)
                fake_logits = self.discriminator(fake_augmented_images, training=True)[0]
                real_logits, gp = self.gradient_penalty(real_augmented_images)          

                d_loss = self.d_loss(real_logits, fake_logits)
                d_total = d_loss + gp 
                consistency_loss = tf.constant(0.0, dtype=tf.float32)

        d_gradients = d_tape.gradient(d_total, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_weights)
        )

        noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
        # train the generator 
        with tf.GradientTape() as g_tape:
            if self.bcr:
                fake_images = self.generator(noise, training=True)[0]
                fake_logits = self.discriminator(fake_images, training=True)[0]
                g_loss = self.g_loss(fake_logits)
            else:
                fake_images = self.generator(noise, training=True)[0]
                fake_augmented_images = DiffAugment(fake_images, policy=self.policy) 
                fake_logits = self.discriminator(fake_augmented_images, training=True)[0]
                g_loss = self.g_loss(fake_logits)
            
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_weights)
        )
        for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
            ema_weight.assign(self.ema_decay * ema_weight + (1 - self.ema_decay) * weight)
            
        update_metrics(
         self.train_metrics,
         g_loss=g_loss,
         d_loss=d_loss,
         gp=gp,
         d_total=d_total,
         real_acc=tf.reduce_mean(real_logits),
         fake_acc=tf.reduce_mean(fake_logits),
         cr=consistency_loss
      )
        
    def create_ckpt(self, model_dir, max_ckpt_to_keep, restore_best=True):
        # log dir
        self.model_dir = model_dir
        log_dir = os.path.join(model_dir, 'log-dir')
        self.writer = tf.summary.create_file_writer(log_dir)
        
        # checkpoint dir
        checkpoint_dir = os.path.join(model_dir, 'training-checkpoints')
        best_checkpoint_dir = os.path.join(
            model_dir, 'best-training-checkpoints'
        )

        self.ckpt = tf.train.Checkpoint(g_optimizer=self.g_optimizer,
                d_optimizer=self.d_optimizer, generator=self.generator,
                ema_generator=self.ema_generator, 
                discriminator=self.discriminator,
                n_images=tf.Variable(0),
                fid=tf.Variable(10000.0), # initialize with big value
                best_fid=tf.Variable(10000.0), # initialize with big value
        ) 
        
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=checkpoint_dir, 
            max_to_keep=max_ckpt_to_keep
         )
        self.best_ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=best_checkpoint_dir, 
            max_to_keep=max_ckpt_to_keep
        )
               
        if restore_best == True and self.best_ckpt_manager.latest_checkpoint:    
            last_ckpt = self.best_ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Best checkpoint restored from {last_ckpt}')
        elif restore_best == False and self.ckpt_manager.latest_checkpoint:
            last_ckpt = self.ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Checkpoint restored from {last_ckpt}')     
        else:
            print(f'Checkpoint created at {model_dir} dir')
            
    def restore_generator(self, model_dir):
        self.model_dir = model_dir
        log_dir = os.path.join(model_dir, 'log-dir')
        
        # checkpoint dir
        best_checkpoint_dir = os.path.join(
            model_dir, 'best-training-checkpoints'
        )

        self.ckpt = tf.train.Checkpoint(
                ema_generator=self.ema_generator, 
                n_images=tf.Variable(0),
        ) 

        self.best_ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=best_checkpoint_dir, 
            max_to_keep=1
        )
                  
        last_ckpt = self.best_ckpt_manager.latest_checkpoint
        self.ckpt.restore(last_ckpt)
        print(f'Best checkpoint restored from {last_ckpt}')

    def save_metrics(self, n_images): 
        # tensorboard  
        with self.writer.as_default():
            for name, metric in self.train_metrics.items():
                print(f'{name}: {metric.result():.4f} -', end=" ")
                tf.summary.scalar(name, metric.result(), step=n_images)
         # reset metrics        
        reset_metrics(self.train_metrics)
            
    def save_ckpt(self, n_images, n_fid_images, fid_batch_size, test_dir, img_size):
        # fid
        fid = self.fid(n_fid_images, fid_batch_size, test_dir, img_size)
        self.fid_avg.update_state(fid)
        with self.writer.as_default():
            tf.summary.scalar('FID_n_img', self.fid_avg.result(), step=n_images)
            
        # checkpoint
        self.ckpt.n_images.assign(n_images)
        self.ckpt.fid.assign(fid)       
        
        start = time.time()
        if fid < self.ckpt.best_fid:
            self.ckpt.best_fid.assign(fid)
            self.best_ckpt_manager.save(n_images)
            self.ckpt_manager.save(n_images)
            print(f'FID improved. Best checkpoint saved at {n_images} images') 
        else:
            self.ckpt_manager.save(n_images)
            print(f'Checkpoint saved at {n_images} images')  
        print(f'Time for ckpt is {time.time()-start:.4f} sec') 
        
        # reset metrics
        self.fid_avg.reset_states()   
        
    def gen_batches(self, n_images, batch_size, dir_path):
        n_batches = n_images // batch_size
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            noise = tf.random.normal([batch_size, self.noise_dim])
            gen_batch = self.ema_generator(noise, training=False)[0]
            gen_batch = np.clip(deprocess(gen_batch), 0.0, 255)

            img_index = start
            for img in gen_batch:
                img = Image.fromarray(img.astype('uint8'))
                file_name = os.path.join(dir_path, f'{str(img_index)}.png')
                img.save(file_name,"PNG")
                img_index += 1
                
    def fid(self, n_fid_images, batch_size, test_dir, img_size):
        inception = Inception()
        fid_dir = os.path.join(self.model_dir, 'fid')
        os.makedirs(fid_dir, exist_ok=True)
        # fid
        start = time.time()
        print('\nGenerating FID images...') 
        self.gen_batches(n_fid_images, batch_size, fid_dir)
        gen_fid_ds = create_fid_ds(
            fid_dir + '/*.png', batch_size, img_size, n_fid_images
        )
        real_fid_ds = create_fid_ds(
            test_dir, batch_size, img_size, n_fid_images
        )
        m_gen, s_gen = calculate_activation_statistics(
            gen_fid_ds, inception, batch_size
        )
        m_real, s_real = calculate_activation_statistics(
            real_fid_ds, inception, batch_size
        )        
        fid = calculate_frechet_distance(m_real, s_real, m_gen, s_gen)
        print(f'FID: {fid:.4f} - Time for FID score is {time.time()-start:.4f} sec')            
        return fid 
