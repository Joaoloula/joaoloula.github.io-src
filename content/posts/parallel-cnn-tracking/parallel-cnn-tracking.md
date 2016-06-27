Title:  Parallel CNN Tracking
Tuthor: João Loula
Date:   2016-06-27
Category: Math/Programming
Tags: tracking, convolutional neural networks, parallel computing
Slug: parallel-cnn-tracking
publications_src: content/posts/parallel-cnn-tracking/references.bib

Code for this post can be found [here](https://github.com/Joaoloula/siamese-tracking)

# Introduction

The idea of this post is to take the approach described in [@@seebymoving] and implement it in a parallelized fashion. Namely, we will create a Siamese CNNs architecture for object tracking using caffe, and distribute its computations with both coarse and medium-grain parallelization using MPI (for an introduction to neural networks and CNNs, see [these](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) [two](http://colah.github.io/posts/2014-07-Conv-Nets-Modular) posts on Christopher Olah's blog, a sketch of the principles behind Siamese CNNs can be found in my [face-verification post](http://joaoloula.github.io/face-verification.html). Finally, a great introduction to MPI and High Performance Computing in general is Frank Nielsen's book, whose preview can be found [here](https://books.google.fr/books?id=eDiFCwAAQBAJ&pg=PR4&lpg=PR4&dq=ecole+polytechnique+hpc+mpi&source=bl&ots=3vsFSyEWs4&sig=wBI83cR9_-u1PNHlE16ryUDrEgw&hl=en&sa=X&ved=0ahUKEwiZ6dCK_cjNAhUCuBoKHfn2CwEQ6AEIIzAB#v=onepage&q=ecole%20polytechnique%20hpc%20mpi&f=false)).

# Tracking

The goal of this project is to use an architecture called Siamese CNNs to solve an object tracking problem, that is, to map the location of a given object through time in video data, a central problem in areas like autonomous vehicle control and motion-capture videogames.

Siamese CNNs [@@siamese-cnns] are a model consisting of two identical CNNs that share all their weights. We can think of them as embedding two inputs into some highly structured space, where this output can then be used by some other function. Notable examples include using Siamese CNNs to determine, given two photos, whether they represent the same person [@@deepid2] or, given two images taking consecutively by a moving vehicle, determine the translational and rotational movements that the vehicle has performed [@@seebymoving].

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/siamese-cnns.jpg" alt='missing' align='middle' />
	<figcaption> <sup>Visualization of the Siamese CNNs architecture: the two CNNs are identical and share all their weights. In this scheme, their output is directed to an energy function that calculates the norm of the difference (source: [@@siamese-cnns]).</sup>
</figure>



The idea of the implementation is to train the Siamese CNNs model on evenly spaced pairs of frames in a video of an object moving, and to feed their output to another network that will try to learn the object's movement between the two frames.

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/trck2.png" alt='missing' align='middle' />
	<figcaption> <sup>Example of two video frames to serve as input to the Siamese CNNs model: the bounding box represent the ground-truth of the object movement in the dataset (source: [@@car-image]).</sup>
</figure>

# A Sip of Caffe

Caffe [@@caffe] is a deep learning framework written in and interfaced with C++, created by the Berkeley Vision and Learning Center. At its core, it is based on two main objects :

* _Nets_ represent the architecture of the deep neural network : they are comprised of _layers_ of different types (convolutional, fully-connected, dropout etc.) ;

* _Blobs_ are simply C++ arrays : the data structures being passed along the nets.

Blobs are manipulated throughout the net in _forward_ and _backward_ passes : forward passes denote the process in which the neural network takes some data as input and outputs a prediction, while backward passes refer to backpropagation : the comparison of this prediction with the label and the computation of the gradients of the loss function with respect to the parameters throughout the network in a backwards fashion.

# Models

One of the great strengths of Caffe is the fact that its models are stored in plaintext Google Protocol Buffer [@@protobuf] schemas : it is highly serializable and human-readable, and interfaces well with many programming languages (such as C++ and Python). Let's take a look at how to declare a convolution layer in protobuf:

```protobuf 
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
  	name: "conv1_w"
  	lr_mult: 1 
  }
  param {
  	name: "conv1_b"
  	lr_mult: 2 
  }
  convolution_param {
    num_output: 256    
    kernel_size: 5    
    stride: 1          
    weight_filler {
      type: "gaussian" 
      std: 0.01        
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

"Name" and "Type" are very straightforward entries : they define a name and a type for that layer. "Bottom" and "Top" define respectively the input and output of the layer. The "param" section defines rules for the parameters of the layers (weights and biases) : the "name" section will be of utmost importance in this project, since naming the parameters will allow us to share them through networks and thus realize the Siamese CNNs architecture, and "lr\_mult" defines the multipliers of the learning rates for the parameters (making the biases change twice as fast as the weights tends to work well in practice).

# Parallelisation

MPI-Caffe [@@mpi-caffe] is a framework built by a group at the University of Indiana to interface MPI with Caffe. By default it parallelizes all layers of the network through all nodes in the cluster : nodes can be included or excluded from computation in specific layers. Communication processes like MPIBroadcast and MPIGather are written as layers in the .protobuf file, and the framework automatically computes the equivalent expression for the gradients in the backward pass.

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/mpi_caffe.png" alt='missing' align='middle' />
	<figcaption> <sup>Example of a CNN architecture parallelised using MPI-Caffe. The Network is specified on the left, and for each layer there is a "0" when only the root is included in that layer's computation and a "-" when all nodes are included in it. The MPIBroadcast and MPIGather begin and end respectively the parallelised section of the code (source: [@@mpi-caffe]).</sup>
</figure>



One of the great advantages of the model is that possibility of parallelisation is twofold:


* _Across Siamese Networks_ (medium grain): the calculations performed by each of the two Siamese CNNs can be run independently, with their results being sent back to feed the function on top;

* _Across Image Pairs_ (coarse grain): to increase the number of image pairs in each batch in training, and the speed with which they are processed, we can separate them in mini-batches that are processed across different machines in a cluster.

# MNIST

## The Dataset

MNIST [@@mnist] is a dataset consisting of 70,000 28x28 grayscale images (split in a train and a test set in a 6:1 proportion) representing handwritten digits, with labels from 0 to 9 that stand for the digit represented by each image. The dataset is stored in the not-so-intuitive IDX file format, but we'll be using [a CSV version available online](http://pjreddie.com/projects/mnist-in-csv/) in this project.

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/mnist.png" alt='missing' align='middle' />
	<figcaption> <sup>Example of images from the MNIST dataset (source: Rodrigo Benenson's blog).</sup>
</figure>

## Preprocessing

For the tracking task, preprocessing was done by transforming images in the dataset by a combination of rotations and translations. Rotations were restrained to $3°$ intervals in $[-30°, 30°]$, and translations were chosen as integers in $[-3, 3]$.

The task to be learned was posed as classification over the set of possible rotations and translations, with the loss function being the sum of the losses for rotation, x-axis translation and y-axis translation. 

## The Network

Using the nomenclature BCNN (for Base Convolutional Neural Network) for the architecture of the Siamese networks and TCNN (for Top Convolutional Neural Network) for the network that takes input from the Siamese CNNs and outputs the final prediction, the architecture used was the following:


* BCNN :
	* A convolution layer, with 3x3 kernel and 96 filters, followed by ReLU nonlinearity;
	* A 2x2 max-pooling layer;
	* A convolution layer, with 3x3 kernel and 256 filters, followed by ReLU;
	* A 2x2 max-pooling layer;
    
* TCNN :
	* A fully-connected layer, with 500 filters, followed by ReLU nonlinearity;
	* A dropout layer with 0.5 dropout;
	* Three separate fully-connected layers, with 41, 13 and 13 outputs respectively (matching number of rotation, x translation and y translation classes);
	* A softmax layer with logistic loss (with equal weights for each of the three predictions).

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/bcnn-tcnn.jpg" alt='missing' align='middle' />
	<figcaption> <sup>Scheme of a forward pass in the Siamese network: each image in the pair moves through the layers L1, ... Lk in one of the BCNNs, and their output is processed by the TCNN to make the prediction (source: [@@seebymoving]).</sup>
</figure>



# Results

The network was trained using batches of 64 image pairs, with a base learning rate of $10^{-7}$ and inverse decay with $\gamma = 0.1$ and $\text{power}=0.75$. The network seemed to converge after about 1000 iterations, to an accuracy of about $3\%$for the rotation prediction and $14\%$ for the x and y translation predictions (about 1.25 times better than random guessing for the rotation and 2 times better for the translations). 

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/mnist_training.png" alt='missing' align='middle' />
	<figcaption> <sup>Value of the loss function throughout training iterations in the model.</sup>
</figure>

# Coarse-Grain Parallelization

The simplest way to parallelize the program is to run multiple training batches on different nodes, as in the scheme below:

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/mpi_caffe_complete.png" alt='missing' align='middle' />
	<figcaption> <sup>Example of a CNN architecture using fully parallelised using MPI-Caffe (source: [@@mpi-caffe])</sup>.
</figure>



In this case, we're gaining a speedup in the [Gustafson sense](https://en.wikipedia.org/wiki/Gustafson%27s_law), that is, as we raise the number of processors, we also raise the size of the data we can compute in a given time. The speedup expression is then given by:

$$\text{speedup}_{\text{Gustafson}}(P) = \alpha_{seq} + P(1 - \alpha_{seq}) $$

where P is the number of processors and $\alpha_{seq}$ is the proportion of the code that's not being parallelized. Seeing as in this scheme the whole network is being parallelized, we have:

$$\alpha_{seq} \approx 0 \Rightarrow \text{speedup}_{\text{Gustafson}}(P) \approx P $$

Let's see how this fares in practice. In the figure below, we find a comparison of running times for the forward and backward passes in the network for one, two and four cores, the four core option using hyperthreading. What we find is that the two core case follows Gustafson's law closely, with a speedup coefficient of $1.93$. In the four core case, however, performance is no better than with two cores, which probably means that hyperthreading is making no difference for this task.

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/cores_comparison.png" alt='missing' align='middle' />
	<figcaption> <sup>Comparison between forward and backward pass times when running the network with 1, 2 or 4 cores with hyperthreading.</sup>
</figure>

# Medium-Grain Parallelization

The interest of the Siamese CNNs architecture, however, is the possibility of parallelization on a lower level : we can distribute the two BCNN streams to two different nodes in the cluster, and then gather their results to perform the computations on the TCNN. Results are shown in the figure below: we can see that performance is almost as good as in the completely parallelized scheme, which confirms our knowledge that the convolutional layers are by far the most computationally-intensive ones, so that the BCNN accounts for most of the computations in the network. We can also see that the difference between these two parallelization schemes lies almost entirely in the backward pass: we can hypothesize that this is due to increased difficulty in computing the gradient through the gather and broadcast layers in the Medium-Grain scheme. 

<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/siamese-tracking/master/docs/illustrations/parallelizations_comparison.png" alt='missing' align='middle' />
	<figcaption> <sup>Comparison between forward and backward pass times when running the network with no parallelization, with only the BCNN parallelized or with the whole code parallelized .</sup>
</figure>
