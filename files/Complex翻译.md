### 复值深度学习 DoA 估计

AI助手:


AI助手:


AI助手:


AI助手:


AI助手:


用户:
附件中的PDF是从Complex ResNet Aided DoA Estimation for Near-Field MIMO Systems这篇论文中截取的几个部分，请你仔细阅读，帮我生成中文翻译。要求地道并且无AI痕迹
AI助手:


用户:
Make sure to include fileciteturn0file0 in your response to cite this file, or to surface it as a link.<PARSED TEXT FOR PAGE: 1 / 6>arXiv:2007.10590v1 [eess.SP] 21 Jul 2020
1
Complex ResNet Aided DoA Estimation for
Near-Field MIMO Systems
Yashuai Cao, Student Member, IEEE, Tiejun Lv, Senior Member, IEEE
,
Zhipeng Lin, Pingmu Huang, and Fuhong Lin
Abstract—The near-field effect of short-range multiple-input
multiple-output (MIMO) systems imposes many challenges on
direction-of-arrival (DoA) estimation. Most conventional scenar￾ios assume that the far-field planar wavefronts hold. In this
paper, we investigate the DoA estimation problem in short-range
MIMO communications, where the effect of near-field spherical
wave is non-negligible. By converting it into a regression task, a
novel DoA estimation framework based on complex-valued dee
p
learning (CVDL) is proposed for the near-field region in short￾range MIMO communication systems. Under the assumption of
a spherical wave model, the array steering vector is determined
by both the distance and the direction. However, solving thi
s
regression task containing a massive number of variables is
challenging, since datasets need to capture numerous complicated
feature representations. To overcome this, a virtual covariance
matrix (VCM) based on received signals is constructed, and
thus such features extracted from the VCM can deal with the
complicated coupling relationship between the direction and the
distance. Although the emergence of wireless big data driven by
future communication networks promotes deep learning-based
wireless signal processing, the learning algorithms of complex￾valued signals are still ongoing. This paper proposes a one￾dimensional (1-D) residual network that can directly tackle
complex-valued features due to the inherent 1-D structure o
f
signal subspace vectors. In addition, we put forth a cropped VCM
based policy which can be applied to different antenna sizes
.
The proposed method is able to fully exploit the complex-valued
information. Our simulation results demonstrate the superiority
of the proposed CVDL approach over the baseline schemes in
terms of the accuracy of DoA estimation.
I. INTRODUCTION
With the advances of key enabling technologies in the
fifth generation (5G) mobile communications, such as mil￾limeter wave (mmWave), small cell, and massive multiple￾input multiple-output (MIMO), the direction-of-arrival (DoA)
estimation problem [1] in the context of massive MIMO
systems has attracted substantial research in recent years
[2]. In past decades, many DoA estimation methods have
been proposed, varying from subspace-based techniques [3]
–
[5], maximum-likelihood estimators [6], compressed sensing
methods [7], to hybrid methods [8]–[10]. Efficient and accurate
The financial support of the National Natural Science Foundation of China
(NSFC) (Grant No. 61671072 and 61931001) and the Beijing Natural Sci￾ence Foundation (No. L192025) is gratefully acknowledged. (Corresponding
author: Tiejun Lv.
)
Y. Cao, T. Lv, Z. Lin and P. Huang are with the School of Information and
Communication Engineering, Beijing University of Posts and Telecommuni￾cations (BUPT), Beijing 100876, China (e-mail: {yashcao, lvtiejun, linlzp,
pmhuang}@bupt.edu.cn).
F. Lin is with the School of Computer and Communication Engineering,
University of Science and Technology Beijing, Beijing 100083, China (e-mail:
FHLin@ustb.edu.cn).
DoA estimation algorithms are crucial to massive MIMO
systems [11]–[13]. Specifically, accurate DoA information is
shown to be critical for beamforming design in massive MIMO
systems [14].
Most of DoA estimation methods mentioned above assume
far-field planar wavefront. However, this assumption may be
invalid for future mobile systems. As a promising enabling
technique for 5G-and-beyond mobile networks, mmWave mas￾sive MIMO is expected to support high data-rate commu￾nications. However, the short wavelength and large antenna
aperture induce challenges in the modeling of propagation
patterns. When the Rayleigh distance is considered, the last￾order scatterers fall within the near-field region of the receive
antenna, where spherical wave propagation is applicable [15].
On the other hand, the near-field DoA estimation appears to be
vitally important in many other applications, including vehicle￾to-everything (V2X) communications [16], [17], virtual reality,
smart home, and automated driving. Particularly, the DoA
estimation of signals from landmarks is the core technology
of vehicle positioning [18], which is an important branch in
the Internet of Vehicles (IoV).
A. Related Work and Motivation
In near-field DoA estimation, the joint estimation of the
distance and the angle is inevitable, which extends the search
space and increases the search overhead. To overcome this
issue, a subspace method without multidimensional search
is proposed in [19]. These subspace methods require many
snapshots and high signal-to-noise ratio (SNR) to improve
the spatial resolution. Furthermore, a maximum-likelihood￾based approach [20] by iterative estimation is proposed for the
case of a small number of snapshots and low SNR. However,
as the parameter dimension increases, the iterative method
s
inevitably face the predicament of slow processing speed.
Meanwhile, wireless networks are generating a massive
volume of data, from which we can extract useful information
using the state-of-the-art machine learning (ML) techniques
[21]–[24]. There have been some efforts exploring the appli
-
cations of ML in DoA estimation. In [25], the near-field DoA
estimation is modeled as a classification problem based on th
e
support vector machine (SVM) approach. The authors of [26]
extend the application of deep neural networks (DNNs) to a
general far-field acoustic DoA estimation problem, which is
modeled as a classification problem and not suitable for near
-
field estimation. Although these ML-based classification meth￾ods are interesting, their accuracy depends on the quantization<PARSED TEXT FOR PAGE: 2 / 6>2
resolution of angles. In [27], a support vector regression (SVR)
method is proposed for estimating the DoAs of near field
sources. Popular deep learning (DL) methods are also used in
far-field DoA estimation such as in the field of acoustic source
direction finding [28]. However, such models of acoustic DoA
estimation cannot be directly applied to signal processing in
wireless systems. In addition, a real-valued DNN (RVNN) for
estimating DoA in hybrid massive MIMO channels is designed
in [29]. Traditionally, complex-valued signals are split into the
real and imaginary parts, which are real values and fed into
the network [26], [29]–[31]. Nonetheless, this treatment may
fail to capture the correlation between the real part and the
imaginary part, thus incurring the phase information loss.
Most of the existing DL methods focus on the real domain.
However, many wireless communications problems to be
solved are in the complex domain. Merging the conception of
complex-valued neural network (CVNN) with wireless signal
processing may provide a further thrust to enable smart radio.
The complex-valued deep learning (CVDL) methods refer to
the DNNs that can perform complex arithmetic, and the weight
parameters of CVNNs are complex-valued [32], [33]. The
authors of [34]–[36] explored the applications of CVDL in
image and speech processing. These network architectures
are all designed to account for image classification, music
transcription or speech spectrum prediction. Against the above
backdrop, our motivations can be explained from the following
aspects:
• Since the conventional near-field methods require high
SNR and snapshots or online iterative estimation, it is
unlikely to estimate DoA in real time. Moreover, there is a
very limited study about designing DL methods for near￾field DoA estimation. As DL is a powerful tool to solve
intractable nonlinear problems, it would be interesting to
improve the performance robustness if training an end￾to-end network to solve the near-field DoA estimation.
• Both spatial spectrum search and classification-based ML
methods cannot meet the precision requirements when
using coarse grids. Besides, RVNNs may destroy the
structure of complex data via splitting real and imaginary
parts. Thus, a CVNN-based regression model is expected
to address these issues.
• When near-field signals are concerned, the CVDL-based
DoA estimation is not straightforward due to the coupling
between the DoA and range parameters. How to design
input features, activation functions and loss metrics moti￾vates us to pursue a general CVDL-based DoA estimation
framework.
B. Main Contributions
In this paper, a novel CVDL-aided DoA estimation approach
is proposed for near-field MIMO systems, where the DoA
estimation is viewed as an end-to-end regression task rather
than a classification problem. First, to decouple the distance
and the direction, we design a new feature representation based
on the reconstructed virtual covariance matrix (VCM). This
representation can eliminate the learning obstacles in DoA
estimation. Then, a complex-valued residual network (ResNet)
is proposed to learn complex features of the input data.
Compared with the RVNN, the proposed network circumvents
the trivial split and splice between real and imaginary parts.
To the best of our knowledge, it is the first time that the phase
mapping activation function is developed for the CVNN. How
to introduce CVDL into massive MIMO systems has not been
well addressed before.
The main contributions of this paper are summarized as
follows:
• From an ML perspective, we investigate the near-field
DoA estimation problem, which is considered as a phase
feature extraction process of input data. Without the need
for tedious grid-based classification, we train an end-to￾end deep network for angle estimation. This network is
capable of processing complex-valued tasks directly.
• New features, based on the reconstructed VCM instead
of the whole or half covariance matrix, are designed as
the inputs of the CVNN. Therefore, the computational
complexity of our deep networks is greatly decreased.
Moreover, the joint two-dimensional (2-D) parameter
estimation problem in the near-field is transformed into a
one-dimensional (1-D) DoA estimation problem, where
the designed input features are able to eliminate the
complicated coupling relationships between the distance
and the direction.
• A 1-D convolutional complex-valued ResNet is proposed
to process 1-D complex features. We design a residual
block architecture with a shortcut connection to ensure
the training stability. To map the real-valued angles from
the complex-valued output, we devise a phase mapping
activation function specifically for the DoA estimation
task. Besides, we also carefully select different complex￾valued activation functions in the corresponding neuron
layers according to the distribution of the dataset.
• For our specific regression task, we compare the training
results of different loss metrics and determine the suit￾able loss function through comparison. As a result, the
prediction precision is enhanced.
• To realize a general CVNN model which can be appli￾cable to different antenna sizes, we put forth a cropped
VCM based scheme. This scheme can be deployed for
arbitrary antenna size without the need for redesigning
new networks. Moreover, it can reduce the computational
cost since the inputs are cropped to a fixed shape.
The remainder of this paper is organized as follows. In
Section II, we establish the received signal model for the near￾field region. In Section III, the VCM reconstruction method
is proposed to remove the corruption of range parameters on
the signal subspace; and the feature preprocessing scheme is
proposed based on the VCM. In Section IV, we present the
CVDL-aided DoA estimation framework and introduce the
complex ResNet with specific model settings. Our simulation
results are provided in Section V, which demonstrates that the
CVNNs outperform the baseline methods. Our conclusions are
offered in Section VI.
Notations: Lower-case boldface symbols indicate column
vectors, and upper-case boldface symbols refer to matrices;<PARSED TEXT FOR PAGE: 3 / 6>6
!
!
!
!
[R]p,p+t → [R¯ ]p,p+t
!
!
!
!
2
≈ "
M
m=1
!
!
![Rs]m,m
!
!
!
2
#$
(t + 2δp) d sin θm
rm
%2
+
$
κp,mκq,mt(t + 2δp) πd2
λrm
cos2 θm
%2
&
. (22)
Fig. 3. Beam patterns of original and reconstructed covariance
matrices with two signals, which are propagated from −30◦ and 45◦.
matrix Ξ's, i.e.,
span [ξ1, ··· , ξM] = span [a1(θ1), ··· , aM (θM )] , (25)
where span[·] stands for the span of a set of vectors.
IV. PROPOSED DEEP COMPLEX NETWORK
ARCHITECTURE
A. Deep Complex Network Architecture
In signal processing, for the convenience of signal repre￾sentation, we usually deal with complex-valued signals that
contain both the amplitude and phase information. Inspired
by [34] and [35], we design a deep complex network for
meaningful information extraction, which is used to carry out
the DoA estimation.
The proposed complex-valued network is able to simulate
the complex arithmetic operations in an internal mechanism.
Let c denote a complex number in the form of c = u+jv with
u, v ∈ R. Assume that the input of the l-th layer of the network
is a complex vector cl = ul + jvl
, where ul and vl are real
vectors. In the l-th layer of the proposed CVNN model, the
complex weight matrix Wl and complex bias vector bl are
expressed as
Wl = Wl
R + jWl
I, bl = bl
R + jbl
I, (26)
where Wl
R and Wl
I correspond to the real and imaginary parts
of Wl
, respectively; bl
R and bl
I denote the real and imaginary
parts of bl
, respectively.
Empirically, the depth of the network plays an important
role in classification or regression tasks from a training point of
view. However, deeper networks may result in more intractable
issues, such as the gradient exploding or vanishing. Inspired
by the ResNet proposed in [43], we develop a residual block
architecture consisting of several blocks, where a shortcut
connection is added to realize the linear superposition of the
input itself and its nonlinear transformation in each block. The
specific components of each block are shown in Fig. 4.
Input data sequence
MaxPooling
Conv1D Residual Block
MaxPooling
Conv1D Residual Block
Flatten
Affine Layer
Csigmoid + Phase Mapping
shortcut
Padding
Ctanh
Conv1D
Ctanh
Conv1D
Affine Layer
Affine Layer
Output
Fig. 4. The deep complex ResNet model. The network is shown on
the left side, and the right side represents the Conv1D residual block
diagram.
As shown in Fig. 4, the architecture of the proposed com￾plex ResNet is composed of two alternations of convolutional
and pooling layers, a flatten layer, and three affine layers. In
our proposed deep complex network, two 1-D convolution
(Conv1D) layers are contained in a convolutional residual
block. We mainly employ 2 residual blocks to learn the
phase-related features of the input complex-valued data. The
residual block is followed by a pooling layer, which extracts
the critical features. The results after pooling are flattened
into a column and fed into a complex affine layer, whose
outputs are mapped to real-valued phases by specially devised
activation functions. Finally, since we define the near-field
DoA estimation as a regression problem, several real affine
layers must be connected to the end of network to predict the
continuous phase variable. The details of each type of layer
representation are listed in the following.
1) Affine layer: Affine means that the neurons between adja￾cent layers are fully connected. The complex-valued arithmetic
process is expressed as
(
$(Wl
cl
)
%(Wl
cl
)
)
=
(
Wl
R →Wl
I
Wl
I Wl
R
) (ul
vl
)
. (27)
The signal flow diagram of the complex-valued affine layer is
shown in Fig. 5.
2) Convolution layer: The convolution operation of the
CVNN, analogous to affine layer, is expressed as
Wl ! cl =(Wl
R ! ul → Wl
I ! vl
)
+ j(Wl
I ! ul + Wl
R ! vl
). (28)
Additionally, the Conv1D filters are employed in 1-D input
features analysis. If the size of the input matrix is Li × Fi,
where Li is the length of the input vector and Fi is the number<IMAGE FOR PAGE: 3 / 6><PARSED TEXT FOR PAGE: 4 / 6>7
j
-1
u f l
vl f
cl cl+1
Wl
R
Wl
R
Wl
I
bl
R
bl
I
Fig. 5. The schematic model of the affine layer of the proposed
CVNN.
of the feature channels, then the required filter has a dimension
of Lc × Fi × Fc × Sc, where Sc is the stride used to describe
the step size of the convolution operation and Lc is the length
of the filter. In addition, Fc represents the number of filters,
and it determines the number of output channels. A filling-up
convolution method, known as “SAME” padding, is conducted
to avoid losing information from the original data. Then, the
dimension of the output after Conv1D will be 'Li/Sc(×Fc =
Lco×Fc for “SAME” padding. After applying the 1-D pooling
layer with the Lp × Sp kernel, the final size of the output
becomes 'Lco/Sp( × Fc, where Lp denotes the length of the
pooling kernel and Sp is the stride of the pooling kernel.
3) Pooling layer: In convolutional neural networks (CNNs),
the pooling layer is often placed behind the convolution layer
to incorporate the features from small neighborhoods. As a
result, the key features for prediction can be obtained, while
sparse and valid information can be retained. In this paper,
the max-pooling layers are applied to the real and imaginary
parts, respectively.
B. Activation Function and Backpropagation
For CVNNs, the activation function is the most critical part,
since it directly determines the nonlinear fitting ability and the
convergence of the back propagation (BP) algorithms [30].
Generally speaking, complex activation functions should be
analytic and bounded. In other words, the functions should
satisfy the Cauchy-Riemann (C-R) equation. In the complex
analysis, Liouville’s theorem states that complex functions that
are both analytic and bounded must be constant functions
[36]. However, in existing literature, many researchers prefer
bounded functions to analytic functions. Simultaneously, due
to the fact that the phase information is determined by both
the real and the imaginary parts, we consider an activation
function that affects the magnitude to indirectly achieve the
phase learning. To this end, a split complex function is
employed in convolution layers, and it is given by
Ctanh(c) = tanh(u) + j tanh(v). (29)
Although the activation function f(c) = Ctanh(c) is non￾analytic, it is bounded in the complex domain. The bound￾edness of the activation function eliminates the appearance
of singular points. Considering that our training targets have
values in the interval (→π/2, π/2), it will cause a large angle
Predicted DoAs
Subspace vector 1 瀖 Subspace vector M
CVNN CVNN
covariance matrix
crop
Fig. 6. The CVDL based DoA estimation framework.
error in degrees even when the small training loss is reached.
Rescaling from the target values in radians does not work.
Note that the mean of the target data is zero. To make the
model focus more on small differences, we adopt the tanh
function in affine layers since it has the highest slope at the
zero point. Such a design can cause the gradient boosting with
small prediction differences.
We also devise a special activation function for a mapping
from the complex domain to the real domain, thus facilitating
the measurement of prediction errors. To be specific, after
the complex affine layer we define the following activation
functions:
Csigmoid(c) = sigmoid(u) + jsigmoid(v), (30)
ρ = arctan
%(c)
$(c) = arctan $c → c∗
c + c∗
%
, (31)
where ρ is the real-valued phase corresponding to the output
of complex affine layer. To avoid the invalid division in (31),
we employ the sigmoid function in (30) to keep the real part
of the output greater than 0.
In CVNNs, the derivable function L regarding the real and
imaginary parts of parameters can also be applied to BP,
respectively [44]. The derivative processing for the real and
imaginary parts is written as
∂L
∂c = $(∇L(c)) + j%(∇L(c)), (32)
∂L
∂W = $(∇L(c)) $ ∂u
∂WR
+ j ∂u
∂WI
%
+ %(∇L(c))
$ ∂v
∂WR
+ j ∂v
∂WI
%
. (33)
Accordingly, the network parameters Wl
R and Wl
I are iter￾atively updated by means of the stochastic gradient descent
(SGD) algorithm [30].
C. Training Strategy
We intend to propose a strategy which can be applied to
different numbers of antennas at the receiver array, instead of
training special DNNs with different input dimensions corre￾sponding to various numbers of antenna elements. Different<IMAGE FOR PAGE: 4 / 6><PARSED TEXT FOR PAGE: 5 / 6>8
from the existing methods, we design new input features for
a general deep network in our DoA estimation framework.
The training strategy can be seen from Fig. 6. Above all, the
VCM derived from any N-element antenna is cropped to a
fixed size so as to retain the key features. The cropped VCM
is constructed as
R =
*
R'
+
N−Nin 2 +1:N− N−Nin 2 +1,
N−Nin 2 +1:N− N−Nin 2 +1
= ΞsΛsΞH
s + ΞzΛzΞH
z , (34)
where Nin × Nin(Nin < N) is the cropped block size of
*
the original VCM. Given the signal subspace vectors Ξs =
ξ1, ··· , ξM
+
, the input features are constructed as
cinput = ξm ∈ CNin , m = 1, 2, ··· ,M. (35)
Finally, independent subspace vectors are adopted as training
samples to perform the learning task.
Algorithm 1 The proposed CVDL-aided DoA estimation
method
Input:
The received signals y(k), k = 1, 2, ··· , K;
The number of near-field sources M;
Output:
The estimates of DoA, θˆ;
1: Calculate the covariance matrix R based on the samples y(k)
according to (10);
2: Reconstruct the VCM R! according to
[R! ]p,p+t = {[R]δl (t),δl(−t) + [R]χr (t),χr(−t)}/2;
3: Crop the VCM R! obtained by (23) to acquire the compressed
signal subspace Ξs based on (34);
4: Devise the input feature representations cinput based on (35) to
build the training samples;
5: Feed the data into the proposed CVNN depicted in Fig. 4, and
perform forward propagation to obtain the network output θˆ;
6: return θˆ;
In this paper, more attention is paid to angle prediction in
regression tasks. Both mean squared error (MSE) and mean
absolute error (MAE) are popularly used for regression loss
metrics [45]. Although the MSE loss is easy to compute the
gradient, large errors have relatively greater influence on the
MSE than the small errors due to the square. Compared with
the MSE, the MAE loss is more robust to outliers since it
does not use square. Moreover, the MAE is more useful if we
are concerned about small errors such as subtle radian errors.
Therefore, a typical MAE loss function is defined as
L(θˆ, θ) = 1
BM
"
B
b=1
*θˆb → θb*1, (36)
where B is the total number of data samples. The true and
estimated DoAs for the b-th sample are denoted by θb and θˆb,
respectively.
For the purpose of extending to multi-source DoA estima￾tion, we employ a training strategy that only uses the signal
subspace vector as training samples. As such, the subspace
vectors associated with multi-DoA are fed into the network
Table I: CVNN Hyperparameters
Layer Name Configuration
Input Nin × 1
Residual Block 1 Conv1D ×2 3 × 1 × 8 × 1
Conv1D ×2 3 × 8 × 8 × 1
MaxPooling 2 × 2
Residual Block 2 Conv1D ×2 3 × 8 × 4 × 1
Conv1D ×2 3 × 4 × 4 × 1
MaxPooling 2 × 2
Flatten Nflat
Affine (Csigmoid + Phase Mapping) 20
Affine 10
Affine 10
Output 1
Table II: RVNN (TDNN [30]) Architecture
Layer Name Layer Specification
input layer input dim = N#
in
hidden layer 1 context size = 5, filter num = 8
hidden layer 2 context size = 5, filter num = 8
hidden layer 3 context size = 5, filter num = 4
hidden layer 4 context size = 5, filter num = 2
hidden layer 5 context size = 5, filter num = 1
dense layer output dim = 10
dense layer output dim = 10
output layer output dim = 1
separately to train and predict. The proposed CVDL-aided
DoA estimation method is summarised in Algorithm 1.
D. Complexity Analysis
We use the total number of floating-point operations
(FLOPs) to measure the computational complexity of the
proposed network. The computational complexities of both
the convolutional and dense layer operations are 4 times more
expensive than their real counterparts but with the same order
[35]. According to [46], in real-valued networks, the FLOPs
of a conv1D layer is given by 2Li(FiL2
c+1)Fc and the FLOPs
of a dense layer is given by (2I →1)O, where I and O denote
the input and output dimensions, respectively.
Regarding the deep complex model, the hyperparameters of
the CVNN are detailed in Table I. These parameters are listed
according to the order of the layers depicted in Fig. 4, and
the network configuration Lc × Fi × Fc × Sc corresponds to
the numerical expression of the Conv1D kernels in the third
column of Table I. For the training and test sets, each sample of
the dataset contains Nin entries taken from the signal subspace
Ξs generated by using (34). The number of neurons in the
flatten layer is denoted as Nflat.
The time delay neural network (TDNN) architecture pro￾posed in [30] has been taken as the RVNN-based baseline
scheme for regression prediction, and its structure and param￾eter settings are shown in Table II. Since the RVNN cannot
directly deal with the complex-valued features, according to<PARSED TEXT FOR PAGE: 6 / 6>9
[30], the real and imaginary parts of (35) are concatenated as
the input tensor, i.e.,
uinput =
(
$(cinput)
%(cinput)
)
∈ RN"
in , (37)
where N#
in = 2×Nin indicates that the input size of the RVNN
is twice as the input size of the CVNN.
For the case of Nin = 33, the total number of FLOPs of the
proposed CVNN is about 0.24 million. In this case, we have
N#
in = 66 for the real TDNN and its total number of FLOPs
is about 0.34 million. Note that the proposed CVNN can
keep operations within complex structures while conventional
RVNNs achieve brute-force fitting by increasing network depth
and dimensions.
Table III: Parameters of the Near-Field Source Dataset
Parameters Value and Range
Training Set Test Set
Distance (in λ) (200, 1800) step: 25 (200, 1800) step: 25
Direction (◦) (-90, 90) step: 0.01 (-90, 90) step: 0.1
Snapshots 100 100
SNR (dB) 10 10
V. SIMULATION RESULTS AND DISCUSSIONS
A. Simulation Setup
In this section, the CVDL-aided DoA estimation approach is
evaluated by numerous simulations. All parameter settings are
in accordance with the near-field region. The implementation
of our deep network is based on TensorFlow [47]. The
estimation results are compared with the MUSIC method in
[19], the SVR method in [27] and the real TDNN scheme in
[30]. The root mean square error (RMSE) is used to evaluate
our proposed model.
To elaborate a little further, the transmitted signals with
carrier frequency of fc = 28 GHz are used. The corresponding
wavelength is λ = 0.0107 m. We consider a ULA with
N = 65 array elements, thus the position of the reference
array element is nc = 33. We set the array spacing as d = λ/2.
Then, the array aperture is D = 32λ, and the Fresnel region
is between 112λ and 2048λ [41].
For the dataset, the received signals are collected by using
(9), and the input datasets are generated according to the
scheme proposed in Section III-B. In attempting to capture the
impact of multiple factors, the dataset should cover a variety
of conditions, such as snapshots, directions and distance from
the reference array element, which are summarized in Table
III. Note that we train the CVNN offline by employing the
raw data with additive noise as the training samples, where
the SNR and the number of snapshots are fixed at 10 dB
and 100, respectively. Then we will test the performance of
online deployment under various SNRs and snapshots. Based
on our CVNN framework, the learning model of multi-source
scenarios can be trained via adding subspace vector samples
associated with multi-DoA. As an initial work and for the sake
of simplicity, we take M = 1 as an example.
(a)
(b)
Fig. 7. Model training results for different loss functions. (a) Learning
curve of 200 epochs in the offline training phase. (b) Comparing test
errors trained with different loss metrics.
During the training process, the SGD method is employed
to speed up the training, and the adaptive moment estimation
(Adam) algorithm [48] is used to update the parameters. To
investigate the loss functions that are appropriate for our
regression predictive modeling problems, Fig. 7 analyzes the
effects of different loss metrics (i.e., MAE and MSE) on
learning behavior and test performance. Fig. 7(a) shows that
both models converge to a steady state and the MSE loss
are declining faster than the MAE loss. This is because the
gradient of the MSE loss is high for larger loss values at initial
epochs. The spikes appearing in the learning curve are the
unavoidable consequences of the mini-batch GD methods. We
plot Fig. 7(b) to compare the test errors (in rad) trained with
the MAE and MSE loss metrics. It is seen that the MAE￾based test error can reach far below 0.05 after convergence,
but the MSE-based test error is above 0.1. This validates the
effectiveness and suitability of the MAE loss for our regression
predictive task.
In addition, we also study the impacts of residual blocks on
the whole network performance. We train different networks<IMAGE FOR PAGE: 6 / 6>
用户:

Remember you have access to rendered images of pages from the files, in addition to their parsed text content. Some files may include images, while others may not.
These may contain embedded text in images, infographics, complex tables, diagrams, and other visual elements not fully captured in the parsed text.
You can directly view these images to read text within images, interpret infographics, understand diagrams and complex tables. When applicable, combine insights from both parsed text and images to provide comprehensive and enriched answers.

When responding to the user, actively incorporate information from embedded text, images, tables, diagrams, and other visuals in the files when applicable.
Explicitly reference them by mentioning their location (e.g., page number) and summarizing their content. For example:
- "According to the *chart on page 5*, sales have increased by 20% over ..."
- "The *diagram on page 3* illustrates the workflow, highlighting ..."

用户:
All the files uploaded by the user have been fully loaded. Searching won't provide additional information.
AI助手:


用户:

用户:

用户:

AI助手:
下面是对你上传的PDF截图内容的中文翻译。我按论文原有结构整理，并尽量保持学术表达自然、干净，不做生硬直译。该PDF对应论文*ComplexResNetAidedDoAEstimationforNear-FieldMIMOSystems*，题名与作者信息可由arXiv记录核对。译文依据你上传的截取页面整理。

---

#复数ResNet辅助的近场MIMO系统DoA估计

YashuaiCao，TiejunLv，ZhipengLin，PingmuHuang，FuhongLin

##摘要

短距离多输入多输出（MIMO）系统中的近场效应给到达方向（DoA）估计带来了诸多挑战。传统方法大多默认信号满足远场平面波前假设。本文研究短距离MIMO通信中的DoA估计问题，在这类场景下，近场球面波效应不可忽略。我们将该问题转化为一个回归任务，并提出一种面向短距离MIMO通信近场区域的、基于复值深度学习（CVDL）的新型DoA估计框架。

在球面波模型假设下，阵列导向向量不仅与方向有关，也与距离有关。然而，这一回归任务包含大量变量，求解难度很高，因为数据集必须能够表征复杂多样的特征。为解决这一问题，本文基于接收信号构造了一种虚拟协方差矩阵（VCM），从VCM中提取的特征能够处理方向与距离之间复杂的耦合关系。未来通信网络所催生的无线大数据推动了基于深度学习的无线信号处理发展，但针对复值信号的学习算法仍在持续探索中。

为此，本文提出一种一维残差网络。由于信号子空间向量天然具有一维结构，该网络能够直接处理复值特征。此外，我们还提出一种基于裁剪VCM的策略，使该方法能够适配不同规模的天线阵列。所提方法可以充分利用复值信息。仿真结果表明，在DoA估计精度方面，本文提出的CVDL方法优于基线方案。

---

#I.引言

随着第五代（5G）移动通信关键使能技术的发展，例如毫米波（mmWave）、小区密集化以及大规模多输入多输出（massiveMIMO），面向大规模MIMO系统的到达方向（DoA）估计问题近年来受到了广泛关注。过去几十年中，研究者已经提出了多种DoA估计方法，包括基于子空间的技术、最大似然估计、压缩感知方法以及混合型方法等。高效而准确的DoA估计算法对于大规模MIMO系统至关重要。尤其是，精确的DoA信息已被证明是大规模MIMO波束成形设计中的关键前提。

上述大多数DoA估计方法均建立在远场平面波前假设之上。然而，这一假设在未来移动通信系统中可能并不成立。作为5G及后续移动网络的重要支撑技术，毫米波大规模MIMO被寄予支持高速率通信的厚望，但其短波长和大阵列孔径也给传播建模带来了挑战。当考虑瑞利距离时，末级散射体可能落入接收天线的近场区域，此时更适合采用球面波传播模型。

另一方面，近场DoA估计在许多应用中同样十分关键，包括车联网通信、虚拟现实、智能家居以及自动驾驶等。尤其是，来自地标信号的DoA估计是车辆定位中的核心技术，而车辆定位又是车联网中的一个重要分支。

---

##A.相关工作与研究动机

在近场DoA估计中，距离和角度的联合估计不可避免，这会扩展搜索空间并增加搜索开销。为了解决这一问题，文献中提出了无需多维搜索的子空间方法。但这类方法通常需要大量快拍数和较高信噪比，才能获得较好的空间分辨率。此外，也有研究提出基于最大似然的迭代估计方法，以适应少快拍和低信噪比场景。然而，随着参数维度增加，迭代方法不可避免地会面临处理速度缓慢的问题。

与此同时，无线网络正在产生海量数据，而利用先进的机器学习（ML）技术可以从中挖掘有用信息。已有一些工作尝试将机器学习应用于DoA估计。例如，有研究将近场DoA估计建模为分类问题，并采用支持向量机（SVM）方法求解；还有研究将深度神经网络（DNN）推广到一般远场声学DoA估计问题中，但其分类建模方式并不适用于近场估计。尽管这类基于机器学习的分类方法具有一定启发意义，但它们的精度依赖于角度量化分辨率。另有研究提出支持向量回归（SVR）方法来估计近场信源的DoA。深度学习方法也已被用于远场DoA估计，例如声源定位领域。然而，这些面向声学DoA估计的模型并不能直接用于无线系统中的信号处理。

此外，也有文献设计了用于混合大规模MIMO信道DoA估计的实值深度神经网络（RVNN）。传统做法通常是将复值信号拆分为实部和虚部，再作为实值输入送入网络。然而，这种处理方式可能无法刻画实部与虚部之间的相关性，从而导致相位信息损失。

现有大多数深度学习方法主要面向实值域，而无线通信中许多待解决的问题本质上都处于复值域。若能将复值神经网络（CVNN）的思想与无线信号处理相结合，有望进一步推动智能无线电的发展。所谓复值深度学习（CVDL），是指能够执行复数运算、且网络权重参数也是复值的深度神经网络。已有研究探索了CVDL在图像处理和语音处理中的应用，但这些网络结构主要是为图像分类、音乐转录或语音频谱预测而设计的。

基于上述背景，本文的研究动机主要体现在以下几个方面：

1.传统近场方法通常依赖较高信噪比、大量快拍，或需要在线迭代估计，因此难以实现实时DoA估计。而面向近场DoA估计的深度学习方法研究仍较为有限。鉴于深度学习在求解复杂非线性问题上的优势，构建端到端网络来解决近场DoA估计问题，并提升方法的鲁棒性，是一个很有意义的方向。
2.无论是空间谱搜索方法，还是基于分类的机器学习方法，在采用粗网格时都难以满足精度要求。此外，RVNN通过拆分实部与虚部，可能破坏复值数据的结构。因此，基于CVNN的回归模型有望克服这些问题。
3.在近场信号场景下，基于CVDL的DoA估计并非直观可得，因为DoA参数与距离参数之间存在耦合。如何设计输入特征、激活函数以及损失度量，是推动本文构建通用CVDL-DoA估计框架的核心原因。

---

##B.主要贡献

本文提出一种面向近场MIMO系统的、基于CVDL的新型DoA估计方法，并将DoA估计视为端到端回归任务，而非分类问题。首先，为了解耦距离与方向信息，我们设计了一种基于重构虚拟协方差矩阵（VCM）的新特征表示。该表示能够消除DoA学习过程中的主要障碍。随后，本文提出复值残差网络（ResNet）来学习输入数据中的复值特征。与RVNN相比，该网络避免了实部和虚部之间机械拆分与拼接的问题。据我们所知，本文首次为CVNN设计了相位映射激活函数。如何将CVDL引入大规模MIMO系统，此前尚未得到充分研究。

本文的主要贡献概括如下：

-从机器学习角度出发，研究近场DoA估计问题，并将其视为对输入数据相位特征的提取过程。无需繁琐的网格化分类，我们训练了一个端到端深度网络来直接估计角度。该网络能够直接处理复值任务。
-设计了基于重构VCM的新特征，而不是使用完整协方差矩阵或半协方差矩阵作为CVNN输入，从而显著降低了网络计算复杂度。同时，近场中的二维联合参数估计问题被转化为一维DoA估计问题，所设计输入特征能够消除距离与方向之间复杂的耦合关系。
-提出一种一维卷积复值ResNet，用于处理一维复值特征。我们设计了带捷径连接的残差块结构，以保证训练稳定性。为将复值输出映射为实值角度，又专门设计了适用于DoA估计任务的相位映射激活函数。此外，还根据数据分布特点，在不同神经元层中选用了不同的复值激活函数。
-针对本文的回归任务，比较了不同损失函数的训练结果，并通过实验确定了更合适的损失度量，从而提高了预测精度。
-为实现适用于不同天线规模的通用CVNN模型，提出一种基于裁剪VCM的方案。该方案无需重新设计网络结构即可部署于任意天线规模，同时由于输入被裁剪为固定尺寸，也降低了计算代价。

---

#IV.所提深度复值网络结构

##A.深度复值网络结构

在信号处理中，为了便于表示，通常处理的是同时包含幅度信息和相位信息的复值信号。受相关工作的启发，本文设计了一种深度复值网络，用于提取有意义的信息，并据此实现DoA估计。

所提出的复值网络能够在内部机制中模拟复数算术运算。设复数\(c=u+jv\)，其中\(u,v\in\mathbb{R}\)。假设网络第\(l\)层的输入是复向量\(c^l=u^l+jv^l\)，其中\(u^l\)与\(v^l\)均为实向量。在该层中，复权重矩阵\(W^l\)与复偏置向量\(b^l\)表示为：

\[
W^l=W_R^l+jW_I^l,\quadb^l=b_R^l+jb_I^l
\]

其中，\(W_R^l\)和\(W_I^l\)分别表示\(W^l\)的实部与虚部，\(b_R^l\)和\(b_I^l\)分别表示\(b^l\)的实部与虚部。

从训练角度看，网络深度在分类或回归任务中起着重要作用。但更深的网络也可能带来更棘手的问题，例如梯度爆炸或梯度消失。受ResNet启发，本文构建了一种由多个残差块组成的结构，在每个块中加入捷径连接，从而实现输入本身与其非线性变换的线性叠加。第3页图4展示了每个残差块的具体组成。

如第3页图4所示，所提复值ResNet由两组“卷积层+池化层”交替堆叠而成，之后接一个展平层和三个仿射层。在所提出的深度复值网络中，一个卷积残差块内部包含两个一维卷积（Conv1D）层。我们主要利用两个残差块来学习输入复值数据中的相位相关特征。残差块之后接池化层，用于提取关键特征。池化后的结果被展平成列向量，再送入复值仿射层，其输出再通过专门设计的激活函数映射为实值相位。最后，由于本文将近场DoA估计定义为回归问题，因此网络末端还需连接若干实值仿射层，以预测连续的相位变量。

###1）仿射层

仿射层表示相邻层之间的神经元采用全连接方式连接。其复值运算过程可表示为式（27）。第4页图5给出了所提CVNN中复值仿射层的信号流示意图。

###2）卷积层

CVNN的卷积运算与仿射层类似，可表示为式（28）。此外，在一维输入特征分析中，本文使用一维卷积滤波器。若输入矩阵尺寸为\(L_i\timesF_i\)，其中\(L_i\)为输入向量长度，\(F_i\)为特征通道数，则所需滤波器的尺寸为\(L_c\timesF_i\timesF_c\timesS_c\)。其中，\(S_c\)表示卷积步长，\(L_c\)表示滤波器长度，\(F_c\)表示滤波器个数，也决定输出通道数。

为避免原始数据的信息损失，本文采用补零卷积方式，即“SAME”padding。于是，经过Conv1D后，输出尺寸为\(\lceilL_i/S_c\rceil\timesF_c\)。再经过核大小为\(L_p\timesS_p\)的一维池化层后，最终输出尺寸变为\(\lceilL_{co}/S_p\rceil\timesF_c\)，其中\(L_p\)为池化核长度，\(S_p\)为池化步长。

###3）池化层

在卷积神经网络中，池化层通常置于卷积层之后，以融合局部邻域内的特征。这样既能提取出用于预测的关键信息，又能保留稀疏而有效的特征。本文分别对实部和虚部采用最大池化层。

---

##B.激活函数与反向传播

对于CVNN而言，激活函数是最关键的组成部分，因为它直接决定网络的非线性拟合能力以及反向传播算法的收敛性。一般来说，复值激活函数应同时满足解析性和有界性，也就是说应满足柯西—黎曼方程。但根据复分析中的刘维尔定理，一个既解析又有界的复函数只能是常数函数。因此，在已有文献中，许多研究者更倾向于优先选择有界函数，而不是解析函数。

同时，考虑到相位信息由实部和虚部共同决定，本文采用一种通过调制幅值来间接实现相位学习的激活方式。为此，在卷积层中使用如下分裂式复值函数：

\[
Ctanh(c)=\tanh(u)+j\tanh(v)
\]

虽然该函数并非解析函数，但它在复平面内是有界的。有界性可以避免奇异点的出现。考虑到本文训练目标的取值区间为\((-\pi/2,\pi/2)\)，即使训练损失已经较小，也可能在角度单位下造成较大误差。简单将目标值从弧度缩放并不能解决这一问题。需要注意的是，目标数据的均值为零。为了让模型更加关注小差异，本文在仿射层中采用tanh函数，因为它在零点附近斜率最大，这样有助于在预测偏差较小时增强梯度。

此外，本文还设计了一种专门用于将复值域映射到实值域的激活函数，以便于度量预测误差。具体而言，在复值仿射层之后定义如下激活函数：

\[
Csigmoid(c)=sigmoid(u)+j\,sigmoid(v)
\]

并定义实值相位为：

\[
\rho=\arctan\frac{\Im(c)}{\Re(c)}
\]

其中，\(\rho\)表示复值仿射层输出所对应的实值相位。为了避免该式中出现无效除法，本文首先使用sigmoid函数，保证输出实部大于0。

在CVNN中，只要损失函数\(L\)关于参数实部和虚部可导，就可以分别对二者执行反向传播。于是，网络参数\(W_R^l\)和\(W_I^l\)可以通过随机梯度下降（SGD）算法进行迭代更新。

---

##C.训练策略

本文希望提出一种能够适用于不同接收阵列天线数目的通用策略，而不是针对不同输入维度分别训练专用DNN。为此，我们为DoA估计框架设计了新的输入特征。第4页图6给出了训练策略流程。首先，由任意\(N\)元天线阵列得到的VCM会被裁剪为固定尺寸，以保留关键特征。裁剪后的VCM如式（34）所示，其中\(N_{in}\timesN_{in}\)（且\(N_{in}<N\)）表示从原始VCM中截取出的子块尺寸。

在给定信号子空间向量\(\Xi_s=[\xi_1,\cdots,\xi_M]\)的条件下，输入特征被构造为：

\[
c_{input}=\xi_m\in\mathbb{C}^{N_{in}},\quadm=1,2,\cdots,M
\]

最后，采用彼此独立的子空间向量作为训练样本执行学习任务。

###算法1：所提基于CVDL的DoA估计方法

**输入：**
接收信号\(y(k),\,k=1,2,\cdots,K\)；
近场信源数目\(M\)

**输出：**
DoA估计值\(\hat{\theta}\)

1.根据式（10），利用样本\(y(k)\)计算协方差矩阵\(R\)。
2.按照
\([R']_{p,p+t}=\{[R]_{\delta_l(t),\delta_l(-t)}+[R]_{\chi_r(t),\chi_r(-t)}\}/2\)
重构VCM\(R'\)。
3.对式（23）得到的VCM\(R'\)进行裁剪，并依据式（34）得到压缩后的信号子空间\(\Xi_s\)。
4.根据式（35）构造输入特征表示\(c_{input}\)，形成训练样本。
5.将数据输入图4所示的CVNN中，执行前向传播，得到网络输出\(\hat{\theta}\)。
6.返回\(\hat{\theta}\)。

本文重点关注回归任务中的角度预测。均方误差（MSE）和平均绝对误差（MAE）是两种常见的回归损失函数。MSE虽然便于计算梯度，但由于平方项的存在，较大的误差会在损失中占据更大权重。相比之下，MAE对异常值更鲁棒，因为它不使用平方；而且在我们更关注小误差、尤其是微小弧度误差的场景下，MAE更为合适。因此，本文采用如下MAE损失函数：

\[
L(\hat{\theta},\theta)=\frac{1}{BM}\sum_{b=1}^{B}\left|\hat{\theta}_b-\theta_b\right|
\]

其中，\(B\)为样本总数，\(\theta_b\)与\(\hat{\theta}_b\)分别表示第\(b\)个样本的真实DoA与估计DoA。

为了将方法扩展到多信源DoA估计，本文采用一种仅使用信号子空间向量作为训练样本的策略。也就是说，与多DoA对应的各个子空间向量将分别送入网络进行训练与预测。所提出的CVDL-DoA估计方法整体上总结于算法1。

---

##D.复杂度分析

本文采用浮点运算总次数（FLOPs）来衡量所提网络的计算复杂度。复值网络中的卷积层和全连接层在计算上大约是其实值对应层的4倍，但二者复杂度阶数相同。根据已有文献，实值网络中，一维卷积层的FLOPs可表示为\(2L_i(F_iL_c^2+1)F_c\)，全连接层的FLOPs则为\((2I-1)O\)，其中\(I\)和\(O\)分别表示输入与输出维度。

对于本文的深度复值模型，CVNN的超参数列于第5页表I。这些参数按照图4所示网络层次顺序给出，其中第三列中的卷积核配置\(L_c\timesF_i\timesF_c\timesS_c\)对应Conv1D核的具体数值表达。对于训练集和测试集，数据集中的每个样本都包含由式（34）生成的信号子空间\(\Xi_s\)中的\(N_{in}\)个元素。展平层中的神经元数记为\(N_{flat}\)。

文献[30]中提出的时延神经网络（TDNN）被作为RVNN回归预测的基线模型，其结构与参数设置见第5页表II。由于RVNN不能直接处理复值特征，按照该文做法，需要将式（35）中的实部与虚部拼接为输入张量，即：

\[
u_{input}=
\begin{pmatrix}
\Re(c_{input})\\
\Im(c_{input})
\end{pmatrix}
\in\mathbb{R}^{N'_{in}}
\]

其中，\(N'_{in}=2N_{in}\)，表示RVNN的输入维度是CVNN的两倍。

当\(N_{in}=33\)时，所提CVNN的总FLOPs约为24万次。在相同条件下，实值TDNN有\(N'_{in}=66\)，其总FLOPs约为34万次。也就是说，所提CVNN能够在保持复值结构的同时完成运算，而传统RVNN往往只能通过增加网络深度和维度来进行“暴力拟合”。

---

##表I：CVNN超参数

|层名称|配置|
|---|---|
|输入层|\(N_{in}\times1\)|
|残差块1|Conv1D×2：\(3\times1\times8\times1\)|
||Conv1D×2：\(3\times8\times8\times1\)|
||MaxPooling：\(2\times2\)|
|残差块2|Conv1D×2：\(3\times8\times4\times1\)|
||Conv1D×2：\(3\times4\times4\times1\)|
||MaxPooling：\(2\times2\)|
|Flatten|\(N_{flat}\)|
|仿射层（Csigmoid+相位映射）|20|
|仿射层|10|
|仿射层|10|
|输出层|1|

---

##表II：RVNN（TDNN[30]）结构

|层名称|层配置|
|---|---|
|输入层|inputdim=\(N'_{in}\)|
|隐藏层1|contextsize=5,filternum=8|
|隐藏层2|contextsize=5,filternum=8|
|隐藏层3|contextsize=5,filternum=4|
|隐藏层4|contextsize=5,filternum=2|
|隐藏层5|contextsize=5,filternum=1|
|全连接层|outputdim=10|
|全连接层|outputdim=10|
|输出层|outputdim=1|

---

##表III：近场信源数据集参数

|参数|训练集|测试集|
|---|---|---|
|距离（以\(\lambda\)为单位）|(200,1800)，步长25|(200,1800)，步长25|
|方向（°）|(-90,90)，步长0.01|(-90,90)，步长0.1|
|快拍数|100|100|
|SNR（dB）|10|10|

---

#V.仿真结果与讨论

##A.仿真设置

本节通过大量仿真评估所提基于CVDL的DoA估计方法。所有参数设置均符合近场区域条件。深度网络的实现基于TensorFlow。本文将所提方法与文献[19]中的MUSIC方法、文献[27]中的SVR方法以及文献[30]中的实值TDNN方案进行比较，并采用均方根误差（RMSE）作为性能评价指标。

进一步地，发射信号载频设为\(f_c=28\)GHz，对应波长\(\lambda=0.0107\)m。考虑一个具有\(N=65\)个阵元的均匀线阵（ULA），其中参考阵元位置为\(n_c=33\)。阵元间距设为\(d=\lambda/2\)。因此，阵列孔径为\(D=32\lambda\)，其菲涅耳区范围为\(112\lambda\)到\(2048\lambda\)。

在数据集构造方面，接收信号依据式（9）采集，输入数据则按照第III-B节提出的方法生成。为尽可能刻画多种因素影响，数据集覆盖了快拍数、方向以及相对参考阵元的距离等多种条件，其具体设置见第5页表III。需要说明的是，CVNN采用离线训练方式，训练样本为加入噪声后的原始数据，其中SNR和快拍数分别固定为10dB和100。之后再测试其在不同SNR和快拍数条件下的在线部署性能。基于本文提出的CVNN框架，多信源场景下的学习模型也可以通过加入与多DoA对应的子空间向量样本来进行训练。作为初步研究并为简化分析，本文以\(M=1\)的单信源情形为例。

在训练过程中，采用SGD方法加速训练，并使用Adam算法更新参数。为考察适用于本文回归预测问题的损失函数，第6页图7对MAE和MSE两种损失函数在学习行为与测试性能上的影响进行了分析。图7(a)表明，两种模型最终都能收敛到稳定状态，但MSE损失下降得更快，这是因为在训练初期，当损失较大时，MSE具有更高的梯度。学习曲线中出现的尖峰则是小批量梯度下降方法不可避免的现象。图7(b)给出了分别采用MAE和MSE训练时的测试误差（单位：弧度）比较。可以看到，基于MAE的测试误差在收敛后明显低于0.05，而基于MSE的测试误差则高于0.1。这说明，对于本文的回归预测任务，MAE损失更有效，也更合适。

此外，本文还研究了残差块对整体网络性能的影响。

---

如果你要，我下一步可以继续帮你做两件事中的一种：
1.把这份译文整理成更像“论文中文精读版”的风格；
2.直接按“中英对照”格式排版，方便你引用或做笔记。

