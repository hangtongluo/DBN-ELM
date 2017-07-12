tic;
clear all
close all

% maxepoch=50;
% maxepoch=20;
maxepoch=20;
% numhid=500; numpen=500; numpen2=500;
numhid=500; numpen=500; numpen2=500;
fprintf(1,'Converting Raw files into Matlab format \n');
% converter;

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

makebatches;
[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);%784-500
restart=1;
rbm;
hidrecbiases=hidbiases;
save mnistvhclassify vishid hidrecbiases visbiases;

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);%500-500
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save mnisthpclassify hidpen penrecbiases hidgenbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);%500-2000
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2;

clear all;
load mnistvhclassify
load mnisthpclassify
load mnisthp2classify
load digittest
load digittrain
clear batchdata batchtargets batchdigittargets
clear testbatchdata testbatchtargets batchdigittargetstest

N1 = length(targets);
%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases]; %(784+1*500)
w2=[hidpen; penrecbiases]; %(500+1*500)
w3=[hidpen2; penrecbiases2];%(500+1*2000)

digitdata = [digitdata ones(N1,1)];

w1probs = 1./(1 + exp(-digitdata*w1)); w1probs = [w1probs  ones(N1,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N1,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
H = w3probs';  %700       60000
% size(H)
%=======================================================================%
%===========================训练过程=====================================%
%=======================================================================%
T = targets'; %1 60000

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
clear digitdata targets digittargets 
clear digitdatatest targetstest digittargetstest 
clear mnistvhclassify vishid hidrecbiases visbiases
clear mnisthpclassify hidpen penrecbiases hidgenbiases
clear mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2

OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
save OutputWeight
%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
clear H;
[I train_output]=max(Y',[],2);%返回最大的预测标签的位置
[I1 train_target]=max(T',[],2);%返回最大的预测标签的位置

%%%%%%%%%% Calculate training classification accuracy
train_accuracy = sum(train_output==train_target) / length(train_target)



%=======================================================================%
%===========================测试过程=====================================%
%=======================================================================%
clear all;
load mnistvhclassify
load mnisthpclassify
load mnisthp2classify
load digittest
load digittrain
load OutputWeight
clear batchdata batchtargets batchdigittargets
clear testbatchdata testbatchtargets batchdigittargetstest

N2 = length(targetstest);
%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases]; %(784+1*500)
w2=[hidpen; penrecbiases]; %(500+1*500)
w3=[hidpen2; penrecbiases2];%(500+1*2000)

digitdatatest = [digitdatatest ones(N2,1)];

w1probs = 1./(1 + exp(-digitdatatest*w1)); w1probs = [w1probs  ones(N2,1)];
w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N2,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
H = w3probs';  %700       10000
% size(H)

TV.T = targetstest'; %1 10000

clear digitdata targets digittargets 
clear digitdatatest targetstest digittargetstest 
clear mnistvhclassify vishid hidrecbiases visbiases
clear mnisthpclassify hidpen penrecbiases hidgenbiases
clear mnisthp2classify hidpen2 penrecbiases2 hidgenbiases2

TY=(H' * OutputWeight)';                       %   TY: the actual output of the testing data
[TI test_output]=max(TY',[],2);%返回最大的预测标签的位置
[TI1 test_target]=max(TV.T',[],2);%返回最大的预测标签的位置

%%%%%%%%%% Calculate testing classification accuracy
test_accuracy = sum(test_output==test_target) / length(test_target)

t=toc



% train_accuracy =
% 
%     0.9464
% 
% 
% test_accuracy =
% 
%     0.9479
% 
% 
% t =
% 
%   218.2175
% 



