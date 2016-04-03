Title: What can CNNs tell us about the visual cortex?
Status: draft



We'll start by using a model pretrained on ImageNet, a large natural image database.
We'll choose the VGG model, created by the Oxford Visual Geometry Group, which consists of 16 layers, of which 5 are convolutions. We'll load the model in theano...

The network is now able to classify natural images: let's see how it fares on a few examples taken from Imagenet:
![alt-text]{filename}cnn_predictions.png


Now, let's take a closer look at the first layer; these are some of the filters it has learned:
{filename}images/cnn_filters.png

If you've done some image processing, you might notice that these are pretty similar to Gabor filters. Little explanation on gabor filters, the fact that they also show up in the visual cortex etc.

Not all of the filters seem to be taking care of finding out orientation and spatial frequency, though. In fact, a lot of filters seem to be looking at the color of an image, like this one:

Image of color filter, perhaps compared to OR/SF one

For this post, we'll be interested mainly in gabor filter-like structures, and so we'll set all the other filters aside for the moment using the hypothesis that a filter in the first layer of a CNN is gabor-like if and only if it doesn't care too much about color (i.e. its R, G and B channels are reasonably similar to one another). By computing the norm of the differences between channels, we arrive at this histogram.

Image of the RGB difference histogram

We'll throw away all the filters whose difference coefficient surpasses 1: that still leaves us with a lot of gabor-like filters to play with.

The first thing we would like to do is evaluate what the preferred orientation and spatial frequency of these filters are. We can do that by projecting them onto gabor filters of varying orientation and spatial frequency, and calculating similarity coefficients from that. This is what the preffered orientation and spatial frequency histograms look like:

Image of the preferred OR and SF histograms

We can now go even further: by fixing the preferred orientation on the gabor filter and varying its spatial-frequency, we can calculate a tuning curve for each CNN filter.

Tuning curves

Lastly, we can try to group this set of filters togethers in a way that minimizes an energy function: we hope from this to recover structure similar to that found in the visual cortex.
