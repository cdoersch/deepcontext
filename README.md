# Unsupervised Representation Learning By Context Prediction

Created by Carl Doersch (Carnegie Mellon / UC Berkeley)

### Introduction
This code is designed to train a visual representation from a raw, unlabeled
image collection.  The resulting representation seems to be useful for standard
vision tasks like object detection, surface normal estimation, and visual data
mining.  

This algorithm was originally described in [Unsupervised Visual Representation
Learning by Context 
Prediction](http://graphics.cs.cmu.edu/projects/deepContext/), which was
presented at ICCV 2015.

This code is significantly refactored from what was used to produce the results
in the paper, and minor modifications have been made.  While I do not expect
these modifications to significantly impact results, I have not yet fully
tested the new codebase, and will need a few more weeks to do so.  
Qualitative behavior early in the training on appears to be equivalent, but 
you should still use this code with caution.

### Citing this codebase
If you find this code useful, please consider citing:

    @inproceedings{doersch2015unsupervised,
        Author = {Doersch, Carl and Gupta, Abhinav and Efros, Alexei A.},
        Title = {Unsupervised Visual Representation Learning by Context Prediction},
        Booktitle = {International Conference on Computer Vision ({ICCV})},
        Year = {2015}
    }

### Installation 

1. Clone the deepcontext repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/cdoersch/deepcontext.git
  ```

2. Build Caffe and pycaffe
    ```Shell
    cd $DEEPCONTEXT_ROOT/caffe_ext
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
   External caffe installations should work as well, but need to be downloaded 
   from Github later than November 22, 2015 to support all required features.

3. Copy `deepcontext_config.py.example` to `deepcontext_config.py`and edit it to
   supply your path to ImageNet, and provide an output directory that the code
   can use for temporary files, including snapshots.

### Running
1. Execute train.py inside python.  This will begin an infinite training loop, which
   snapshots every 2000 iterations.  The results in the paper used a model that
   trained for about 1M iterations.

   By the code will run on GPU 0; you can use the environment variable 
   CUDA_VISIBLE_DEVICES to change the GPU. 

   All testing was done with python 2.7.  It is recommended that you run inside
   ipython using `execfile('train.py').

2. To stop the train.py script, create the file `train_quit` in the directory 
   where you ran the code.  This roundabout approach is required because the 
   code starts background processes to load data, and it's difficult to
   guarantee that these background threads will be terminated if the code is
   interrupted via `Ctrl+C`.

   If train.py is re-started after it is quit, it will examine the output
   directory and attempt to continue from the snapshot with the higest
   iteration number.

3. You can pause the training at any time by creating the file `train_pause` in
   the directory where you ran the code.  This will let you use pycaffe to
   examine the network state.  Re-run train.py to continue.

