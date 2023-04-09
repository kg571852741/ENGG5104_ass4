Environmnets:
    torch  2.0.0
    Python 3.9.16

Completed Task: Task1,Task2,Task3,Task4 and Task5
For the four models being tested (FlowNetE, FlowNetER, FlowNetERM, FlowNetOurs) with inference mode activated, the number of parameters, FLOPs count and the loss function used were as follows:

FlowNetE:

    Number of Parameters: 2822722
    FLOPs Count: 2174.09M
    Loss Function: EPE (End-Point-Error)
    Inference Average: 6.002
    
FlowNetER:

    Number of Parameters: 7015234
    FLOPs Count: 2193.56M
    Loss Function: EPE (End-Point-Error)
    Inference Average: 6.030

FlowNetERM:

    Number of Parameters: 7041934
    FLOPs Count: 2202.55M
    Loss Function: Multiscale
    Inference Average: 5.992

FlowNetOurs:

    Number of Parameters: 10336948
    FLOPs Count: 2259.67M 
    Loss Function: OursLoss (multi-scale loss)
    Inference Average: 5.4778

Note:
    All models have div_flow set to 20, and were tested with the MpiSintelClean dataset for inference.
    The class model does not have the attribute to module, and any instances of "model.module" are replaced with "model" to avoid the AttributeError.
    All training is performed using the CPU.

Task1: Implement FlowNet Encoder
    a) Implement `networks/FlowNetE.py'

Comment:
    The FlowNetEncoder class is a PyTorch module that performs encoding for the FlowNet model. 
    It normalizes the input using the mean and contains several convolutional layers with LeakyReLU activation functions. 
    The output is the flow at the last scale, which is obtained using an upsampling layer. 
    The div_flow parameter is used for training, and the code provides an alternative implementation with different parameters.

Task2: Loss Function
    a) Implement `EPELoss' in `losses.py'
    b) run `run_E.sh'

Comment:
    The code defines the EPELoss class, a PyTorch module for calculating the endpoint error (EPE) loss between predicted and ground-truth flow. 
    It takes two inputs, output and target, which are tensors representing the predicted flow and the ground-truth flow, respectively. 
    The div_flow parameter is used to obtain small output values for easy training and is multiplied by the target tensor. 
    The EPE is calculated as the Euclidean distance between the predicted and ground-truth flows and is then averaged over the batch. 
    The module returns the EPE value as a list.

Task3: Refinement Module
    a) Implement `networks/FlowNetER.py'
    b) run `run_ER.sh'

Comment:
    The code defines the FlowNetEncoderRefine class, a PyTorch module for the encoder part of a refined FlowNet model.
    It takes input in the form of tensors and normalizes it using the mean of the input tensor.
    It contains several convolutional layers with LeakyReLU activation functions, and two deconvolutional layers.
    The output of the model is the flow at scale 2, which is obtained by concatenating the output of the convolutional layers with the output of the deconvolutional layers and passing it through a final convolutional layer followed by an upsampling layer. 
    The div_flow parameter is used to obtain small output values for easy training and is ignored during inference.

    Task4: Multi-scale Optimization
    a) Implement `networks/FlowNetERM.py'
    b) Implement `MultiscaleLoss' in `losses.py'
    c) run `run_ERM.sh'

Commnet:
    The FlowNetEncoderRefine class has 1 scale of flow output while the FlowNetEncoderRefineMultiscale class has 3 scales of flow output. 
    FlowNetEncoderRefineMultiscale also has additional layers for upsampling and concatenation to incorporate multi-scale information. 
    Specifically, it has deconvolutional layers for upsampling and concatenation with the lower-scale feature maps, as well as additional convolutional layers to predict the flow at each scale.
    The output of FlowNetEncoderRefineMultiscale can be either the flow at scale 2 multiplied by div_flow or a tuple of flows at scales 2, 3, and 4 multiplied by div_flow when in training mode.


Task5: Open Challenge
    a) go into the `open_challenge' directory, implement 'networks/FlowNetOurs.py' and 'Oursloss' in 'losses.py'
    b) you can also modify input transformation code in 'dataset.py'
    c) run `run_ours.sh'

Comment : 
   The model contains several convolutional and deconvolutional layers with non-linear activation functions such as LeakyReLU. 
   The forward function normalizes the input data and then passes it through the layers to generate flow estimates at different scales. 
   The function returns flow estimates at scale 2 and 3, and optionally at scale 4 if in training mode. 
   There is another commented implementation of the same model that takes 6 input channels and uses a spatial correlation sampler.
   The performance metrics for the FlowNetOurs deep learning model.
   The first section shows the number of floating-point operations (FLOPs) for each layer in the model, indicating a total of 2259.67M FLOPs. 
   The mdoel has 4.03M trainable parameters and an effective batch size of 8. 
   The loading of a saved checkpoint for the model, which was trained for 41 epochs. 
   The Adam optimizer is then initialized with various parameters before the model is run on some data with an average inference time of 5.746 seconds.

OurLoss:
    OursLoss that inherits from the nn.Module class in PyTorch. 
    It implements the multi-scale loss for estimating the flow between two images, as described in the reference paper cited in the code.
    It uses a weighted sum of L2 norms of the difference between the target and predicted flow at different scales. 

Notation
a) the best validation EPE is printed in the log.
b) FLOPs and params are printed in the log.
c) `test_*.sh' is used to evalute the trained model

Submission
a) you need to sumbit the codes and trained checkpoints (save in `work/' automatically)
b) you should only keep `*_model_best.pth.tar' and clean other irrelevant files in `work/'
c) you should move `data/' out before `zip'
d) add a readme file to tell the TAs which tasks you complete.

