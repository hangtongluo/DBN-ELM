% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.
tic;
clear all
close all

%定义最大的批迭代次数
maxepoch=20; %In the Science paper we use maxepoch=50, but it works just fine.
%设定隐层数量（1000,500,250,30）递进
numhid=1000; numpen=500; numpen2=250; numopen=30;

fprintf(1,'Converting Raw files into Matlab format \n');
converter;

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

makebatches;  %会把训练数据和测试数据存储成固定的格式
[numcases numdims numbatches]=size(batchdata);  %输出训练数据的维度值

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);%28*28-1000
restart=1;
rbm;%通过二元隐层神经元和可视层神经元训练RBM
hidrecbiases=hidbiases;  %第一层RBM学习出的隐层偏置
save mnistvh vishid hidrecbiases visbiases;  %保存第一层RBM的权重、隐层偏置、可视层偏置

%pause

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);%1000-500
batchdata=batchposhidprobs;  %第一层隐层输出数据（降维一次结果）
numhid=numpen;  %隐层数改变为500
restart=1;
rbm;%通过二元隐层神经元和可视层神经元训练RBM
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthp hidpen penrecbiases hidgenbiases;  %保存第二层RBM的权重、隐层偏置、可视层偏置

%pause

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);%500-250
batchdata=batchposhidprobs;  %第二层隐层输出数据（降维二次结果）
numhid=numpen2;%隐层数改变为250
restart=1;
rbm;%通过二元隐层神经元和可视层神经元训练RBM
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2 hidpen2 penrecbiases2 hidgenbiases2;  %保存第三层RBM的权重、隐层偏置、可视层偏置

%pause

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numopen);%250-30
batchdata=batchposhidprobs; %第三层隐层输出数据（降维三次结果）
numhid=numopen;%隐层数改变为30
restart=1;
rbmhidlinear;%通过高斯隐层神经元和可视层二元神经元训练RBM
hidtop=vishid; toprecbiases=hidbiases; topgenbiases=visbiases;
save mnistpo hidtop toprecbiases topgenbiases;

%pause

backprop;   %微调达到相应的系数

t=toc;
