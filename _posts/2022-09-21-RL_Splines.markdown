---
layout: post
title:  "Object detection & Reinforcement Learning-based spline fitting"
subtitle: "My internship at EMBL-EBI in Cambridge at the Uhlmann group"
background: '/img/cells.png'
date:   2022-09-21 14:00:00 +0200
categories : research
published: true
hidden: false
---

During summer 2019 I had the pleasure to do an internship with [Virginie Uhlmann 's team](https://www.ebi.ac.uk/research/uhlmann) at the [European Bioinformatics Institute (EBI)](https://www.ebi.ac.uk) . The team aims at developing tools that blend mathematical models and image processing algorithms to quantitatively characterize the content of bioimages.

In this post I will discuss the methods used during the course of this internship on *multiple cell tracking in time lapse microscopy images*. Most of the methods described in this post were adapted from the cited publications to our case and implemented by me in **Python** and **Tensorflow**.

Goal
====================================

My goal during this 6 months internship was to further develop techniques to track individual _Mycobacteria smegmatis_ cells in sequences of time-lapse microscopy images.
Since these images are challenging because of **division events**, **compact cell colonies** and little prior on individual **cell shapes**, classical segment-then-track methods are ineffective. We relied on a **graphical model** solution to solve the **tracking** and **segmentation** problem jointly at once on the whole sequence. We also need to identify **division events** and thus build a _Mycobacteria smegmatis_ genealogical tree of some sorts.
To build the graphical model, cell **candidates** must be identified in each individual image. To do so, we explored the use of several convolutional neural network models from U-net to discriminative losses for instance segmentation.

## _Mycobacteria smegmatis_
We were working on time lapse images of _Mycobacteria smegmatis_ that is a non pathogenic bacterial species that strongly resembles the _Mycobacterium tuberculosis_, making it a great tool to study pathogens in a safe environment. Several works {% cite Santi2013 Santie01999-14 %} study the replication mechanism of these cells and how it is linked to antibiotics resistance

<figure>
	<center>
  	<img src="{{site.url}}/img/posts/tracking_cells/mycobacteria.png" alt="Mycobacteria smegmatis" style="width:75%"/>
  	<figcaption>Mycobacteria smegmatis in phase microscopy overlayed with fluorescence data {% cite ginda2017studies %}</figcaption>
  	</center>
</figure>


## The data
The data (courtesy of LMIC, EPFL, Lausanne, Switzerland) consist of several time-lapse microscopy videos showing the growth of M. smegmatis colonies. Images are composed of a phase contrast and a fluorescence channel reporting cellular division. In approximately the first 3/4 of the frames of each videos, the medial axis of the bacterias have been annotated with a [custom software](http://bigwww.epfl.ch/teaching/projects/abstracts/mariani/).


## Ambiguity in single frame segmentation
The main issue with this data is that the detection relies mainly on the temporal data since on a given frame it is not obvious to the human eye where the boundaries of the cells are since cells are often cluttered.
<figure>
	<center>
  	<img src="{{site.url}}/img/posts/tracking_cells/detection.png" alt="Detection ambiguity" style="width:100%"/>
  	<figcaption>Example of different detection solutions for a single image (the input image in this figure is a substitute of the raw data)</figcaption>
  	</center>
</figure>
Here a simple input frame with fews cells can lead to multiple hypothesis on the number and placement of cells in the frame. This shows how there is ambiguity on individual cells detection even in a few cell setup. Later in the time-lapse there can be thousands of cells at once. Therefore, because of the in-frame ambiguity, we cannot simply segment each frame and then track between frames.

From that example one can intuit that some solutions are more likely than others, also by using the data from the previous and next frame we may have more insight on the most probable solution. For that matter we used a graphical model to make use of all of the data available in order to solve the detection and tracking.

_________________________________


Graphical models for joint Segmentation and tracking
====================================


Schiegg et al. propose a model based on factor graphs to represent the multiple detection hypothesis and jointly tracking and segmenting on whole videos at once {% cite schiegg2014graphical %}. To put it in a nutshell, given that we provide a set of **detection candidates** and **transition candidates** , along with **associated probabilities** , this method uses a graph representation to compute the most likely candidates for detections and transitions within some **constraints** .

* **detection candidates** are potential cells (for instance the region labeled 1 in the figure, or the union of 2 and 3 labeled 23)
* **transition candidates** represent how these candidates would transition from one frame to the next. Here one could expect 12 (at frame t)to transition into 4 (at frame t+1)
* **constraints** are what makes the solution physically sound, no cell candidates can overlap (cell candidate 12 could not co-exist with 23), cells should not appear and disappear in the middle of the video etc...

For each of the **detection candidates** and **transition candidates** we can provide an associated **probability**, this is provided by a simple classifier that uses as input some cell data (shape, length etc..) to infer its likelihood of being a cell or transition regardless of context. For instance a one pixel cell would have a very low probability and a transition implying that a cell would travel from one end of the image to the other in a single frame is also rated with a low probability.

<figure>
	<center>
  	<img src="{{site.url}}/img/posts/tracking_cells/graphical_model_complete.png" alt="A graphical model" style="width:100%"/>
  	<figcaption>Sample graphical model for only two frames, each <b>detection candidate</b> is represented by a green node. Yellow nodes correspond to <b>transition candidates</b> between frames. <b>Constraints</b> are represented by the square nodes</figcaption>
  	</center>
</figure>

To explain the technical functioning of a graphical model would take another presentation, for this time we will focus on how to generate these segmentation proposals. Also this part of the graphical inference model had already been implemented by C. Haubold {% cite haubold2017scalable %}.


## Graphical model model limitation
From the previous example we can expect the graph to grow exponentially as the number cell proposal augments. Thus it is important to generate a set of segmentation proposals that is wide enough to contain the ground truth but small enough not to make the complexity explode.


Previous approach
====================================
*BactImAS* {% cite mekterovic2014bactimas %} offers a semi-automated solution to the tracking problem where the user must manually define initial cells, division events, and maybe correct the model's prediction in between. To detect cells *BactImAS* relies on edge detection, thresholds and skeletonization.
In her PhD thesis {% cite uhlmann2017landmark %}, Virginie Uhlman introduces splines to further automate the process. From the detected cell tips a set of shortest viable path in between are generated as cell proposals. The issue of the exploding graph complexity remains because of the high number of proposals.

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/uhlmann-prev-appr.jpg" alt="Uhlmann previous approach" style="width:100%"/>
    <figcaption>V. Uhlmann's approach. 1 : identify cell tips with pixel classifier, 2 and 3 : model all possible tip to tip links as a shortest path relying on splines {% cite uhlmann2017landmark %}</figcaption>
    </center>
</figure>

_________________________________


Exploring candidates generation through deep learning
====================================

The aim of my internship project was to explore the generation of cell candidates using deep learning methods. The idea being that a number of cell proposal could be ruled out by their improbable shape, thus adding a learning component (opposed to doing only image processing) to the process could improve the proposals.

## First approach : U-net for pixel class prediction

Graphical models provide a way to solve for the tracking and segmentation problem simultaneously on the entire video. However, it relies on cell candidates generated for each frame. Since we want the true segmentation to be within the set of candidates, we perform **over-segmentation** (i.e. we segment too much) to ensure that we obtain more false positives that false negatives (for that we can rule out false positives with the graphical model).
Deep neural networks can be used to generate such segmentation proposals in a robust and generalisable way.
For instance, {% cite Falk2019 %} propose a deep learning method for detecting instances of cells in images relying on pixel-wise classification. For that matter we used a U-Net architecture.

### What is a U-net ?
**U-net** is a convolutional neural network first introduced by {% cite RonnebergerFB15 %} for biomedical image segmentation. It aims at providing fine pixel classification with low and high scale awareness of the neighboring pixels. The network is composed of a contracting path that decreases the image size through **convolution** and **pooling** while increasing the number of channels, and an expansive path that increases the image size through **convolution** and **up-sampling** while decreasing the number of channels. The output image scale  is therefore the same as the input. **Skip paths** are also added to link layers  within levels, so that the information capturing fine details can be forwarded to the end of the network without traversing the contracting path.

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/u-net-architecture.png" alt="Unet Architecture" style="width:100%"/>
    <figcaption>Structure of a basic U-net {% cite RonnebergerFB15 %}. The <b>skip paths</b> are shown in gray</figcaption>
    </center>
</figure>


### Weighted soft-max cross-entropy loss

We use a Cross entropy loss with soft-max as it is commonly used for classification {% cite Bishop:2006:PRM:1162264 %} and can be written as

$$
l(I) := - \sum_{x\in I}w(x) \log
\frac{
\exp \left(    \hat{y}_{y(x)}(x)  \right)
}{
\sum_{k=0}^K\exp \left(    \hat{y}_{k}(x)  \right)
},
$$

with $$x$$ a pixel in the image domain $$I$$, $$\hat{y}_k:I\rightarrow\mathbb{R}$$ the predicted score for class $$k$$, $$K$$ the number of classes, and $$ y:I\rightarrow\{0,\dots,K\}$$ the ground truth segmentation. Thus, $$\hat{y}_{y(x)}(x)$$ corresponds to the predicted score for ground-truth class $$y(x)$$ at position $$x$$. The $$w(x)$$ is a per-pixel weighting term used to handle class imbalance and instance separation, and is defined as

$$
    w:=w_\mathrm{bal} +w_\mathrm{sep}
$$

#### Learning instance segmentation

To learn how to segment instances, labels of different instances must be separated by at least one pixel of background, ensuring that each instance is one single connected component. In order for this gap to be predicted correctly by the network, a weighting term $$w_\mathrm{sep}$$ is applied on the loss to further penalize errors in boundary areas as

$$
    w_\mathrm{sep}(x) := \exp \left( -\frac{(d_1(x)+d_2(x)}{2\sigma^2}\right),
$$

with $$d_1$$ and $$d_2$$ the distances to the two nearest instances. This weight therefore increases at locations where two cells are close together, enforcing a separation. The next figure shows an illustration of  $$w_\mathrm{sep}(x)$$ over an image.

#### Pixel class prediction

Our first objective is to distinguish cells from background, thus we use a background class and a cell class. However individual cells need to be spatially separated by a gap in order to distinguish different entities, hence we introduce an **inner cell** and **outer cell** class in order to have separate cell cores between and also not to confuse the classifier by labeling the outer cells as background. Since the cell tips will be a useful information to separate individual cells afterwards we also introduce a **cell tip** class.

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/unet-train-weighted.png" alt="Unet training" style="width:100%"/>
    <figcaption>Training the U-net with our data.</figcaption>
    </center>
</figure>

Once the U-net is trained, the network outputs probability maps for each class.

### Candidates generation

From output of the network we need to produce a set of cell candidates. We devised a method of _over-segmentation_ to get a set of cell parts that could be assembled to make cell candidates. Basically, from the separated **inner cells**, we run a watershed algorithm to propagate labels on the cells. We get the results shown below on a sample image. From the over-segmentation output we can build a set of cell proposals. We build proposals from the cell segments by proposing every segment and every union of touching segments (within a set limit amount, so that cells aren't too big).

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/candidates.png" alt="Candidates generation" style="width:100%"/>
    <figcaption>Building candidates from the predicted pixel classes. Left : the predicted pixel classes. Center : cell segments built from the pixel classification. Right : set of cell candidates built from the available cell segments.</figcaption>
    </center>
</figure>

### Results
This pixel wise classification approach provided decent results, however the candidates generation part relies heavily on **manually chosen hyper-parameters**. Also this methods tends to generate quite a lot of very small cell proposals. These numerous false positive make the graphical model complexity explode. Any attempt at choosing a set of hyper-parameters to reduce the number of false positives dramatically increase the number of false negatives, meaning that the graphical model cannot be set up.

_________________________________

## Instance segmentation with pixel embedding
In the first approach the instance segmentation (differentiating one cells from its neighbor) is done by the proxy of **per pixel classification**. What if we could directly optimize for **instance segmentation** within the convolutional network and thus minimize the handcrafting of the cell proposal method ?

We thus focus on a framework in which the instance segmentation problem becomes a **pixel clustering problem** in a new feature space {%cite deBrabandere2017semantic %}. This approach allows predicting instance segmentation with no prior on the number of elements in the image. In constrast to other popular instance segmentation methods like Mask R-CNN {%cite he2017mask %}, it does not rely on region proposal followed by classification. Instead of doing pixel classification with a softmax loss, which would limit the number of instances, we can detect any number of instances in an image can be captured.
Indeed, when predicting instances as one class for each instance, the number of instances is limited by the size of the output vector, i.e. the number of classes. Also
{%cite payer2018instance %} extend this idea by considering a tracking component to the problem, adding recurrent components in the network and using a new similarity measure in the feature space.

In this framework each pixel is attributed a **N dimensional vector**, so that every pixel from a same cell has a "similar" vector and pixels from different cells have "non-similar" vectors (the similarity measure may depend on the method used). If N=3 you can consider these vectors to be RGB colors, the objective being so that each individual cell is colored uniformly, but different cells have different colors. Through the different figures we project our N dimensional vectors to a 3D space with a PCA so that we can display these vectors as colors. Using the first two dimensions of the PCA we can plot these pixels as points within this 2D space to better visualize the clustering.

### Semantic InstanceSegmentation with a Discriminative Loss Function

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/SIS-DLF-1.jpg" style="width:100%"/>
    <img src="{{site.url}}/img/posts/tracking_cells/SIS-DLF-2.jpg" style="width:100%"/>
    <figcaption>Top left: the input image. Top right: pixel embeddings projected in a 2D space each car is a cluster. Bottom right: pixel embeddings interpreted as colors, each individual car is colored/projected differently. Bottom right: predicted instances after clustering of the pixel embeddings {% cite deBrabandere2017semantic %}</figcaption>
    </center>
</figure>

#### discriminative loss
To learn the clustering, we rely on a discriminative loss composed of three key parts {% cite deBrabandere2017semantic %}. During learning we use as input the image along with ground truth instance labels.

- **Variance term**: is an intra-cluster pull force moving embeddings toward the mean of each label cluster. A margin is set for the variance, which corresponds to the inner circle in the next figure. This margin defines how tightly packed clusters should be.
- **Distance term**: an inter-cluster push force drawing the mean of embeddings of different instances further
apart from each other. A margin is set for the distance, corresponding to the outer
circle in the next figure. This margin defines how far from each other clusters should be.
- **Regularization term**: it is a pull force drawing clusters closer to the origin of the embedding space to avoid values blowing up.

These three part as summed as the loss function and minimized during the learning phase.

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/SIS-DLF-3.jpg" style="width:100%"/>
    <figcaption> {% cite deBrabandere2017semantic %}</figcaption>
    </center>
</figure>

### Clustering
For that matter, we used the agglomerative clustering available in sklearn
{%cite scikit-learn %}. This method starts by initializing each pixel as a cluster. Clusters are
then merged together in a tree-like manner. The merging stops when a distance threshold is met,
that is when the distance between all clusters is more than the threshold. This is relatable to the
design of the discriminative loss since it enforces a minimal distance between different instances. Also this method does not rely on a prior knowledge of the number of clusters.

### Results

Here is the results on a few sample frames. From the second more cluttered frame we observe that this method scales decently to a high number of cells. One improvement might be to add a filter on the smallest size a cluster can be to avoid 1 pixel clusters being proposed as cells.

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/clustering_18_35.png" style="width:100%"/>
    <figcaption> Result on an early frame</figcaption>
    </center>
</figure>

_________________________________________

<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/clustering_18_120.png" style="width:100%"/>
    <figcaption> Result on a more cluttered frame</figcaption>
    </center>
</figure>


## Other approaches and perspectives
We tried implementing a Cosine embedding loss {%cite payer2018instance %} as a replacement for the discriminative loss function. This yielded similar results but is not as straightforward to cluster as instead of using any N dimensional vector, the pixels are embedded in a N dimensional sphere. Meaning that the similarity function for clustering is not longer a simple euclidean distance but rather a cosine similarity. Also {%cite payer2018instance %} proposes to compute the loss over several frame and ensure that cells have a consistent embedding cluster through time, thus learning the **tracking** part along the **instance segmentation** part. Due to time constraints I did not manage to properly test this promising method.

Another idea would be to expand this embedding time consistency to cell divisions, for instance we could devise a loss so that, once projected in a set of specified dimension parent cells would be similar, but remains separate in the other dimensions. If functioning properly this method would negate the need of the graphical model as it solves for **tracking**, **instance segmentation** and **cell divisions** at once.

_________________________________________

Conclusion
====================================
From the different convolutional network methods I used I managed to produce strong sets of cell proposals. However due to time constraints and technical limitation of the graphical model, I did not manage to properly use these cell proposals to train the latter. Meanwhile this project gave me a great insight on the functioning and building on convolutional networks.

Throughout this project, the graphical model was a motivation for our model design choices. A U-net architecture provides an instance segmentation that is not satisfactory as it has extensive post-processing that is specific to the dataset. As we aim for generalizable solution we also explored instance segmentation oriented networks and losses (e.g. discriminative or cosine embedding loss). These offer a more straightforward optimization process to obtain cell candidates for the graphical model. Moreover, the clustering process offers better control on the roughness or finesse of the segmentation candidates.

Unfortunately, the ground truth matching part which is necessary to train the graphical model, proved to be a roadblock when using our first instance predictions obtained using U-net on a pixel classification task. We sadly did not have time to test ground piping our discriminative network results into the graphical model.

_________________________________________

Acknowledgments
====================================

To the French Embassy in London for funding this internship, to EMBL for making it
possible, to José for his technical help,
And mostly, warm and sincere thanks to the whole Uhlmann Lab : Virginie, Soham, Yoann,
Johannes, James and Maria.


_________________________________________


## Bonus: Making some abstract "art" with it

<figure>
  <center>
  <img src="{{site.url}}/img/cells.png" style="width:100%"/>
  </center>
</figure>

As i noticed the network trained on the discriminative loss function outputted some pretty colors I tried to feed it a synthetic image over-packed with cells. This is the result, it makes for a great (but maybe too colorful) wallpaper.
<figure>
  <center>
    <img src="{{site.url}}/img/posts/tracking_cells/super_spagettos.png" style="width:100%"/>
    <figcaption> Left: Input spaghettis. Right: output colors</figcaption>
    </center>
</figure>



_________________________________________


References
====================
