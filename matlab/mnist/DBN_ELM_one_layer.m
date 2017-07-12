tic;
clear all
close all

% maxepoch=50;
% maxepoch=20;
maxepoch=20;
% numhid=500; numpen=500; 
numhid=700; 
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

clear all;
load mnistvhclassify
load digittest
load digittrain
clear batchdata batchtargets batchdigittargets
clear testbatchdata testbatchtargets batchdigittargetstest

N1 = length(targets);
%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases]; %(784+1*500)

digitdata = [digitdata ones(N1,1)];

w1probs = 1./(1 + exp(-digitdata*w1)); 
H = w1probs';  %700       60000
% size(H)
%=======================================================================%
%===========================训练过程=====================================%
%=======================================================================%
T = targets'; %1 60000

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
clear digitdata targets digittargets 
clear digitdatatest targetstest digittargetstest 
clear mnistvhclassify vishid hidrecbiases visbiases

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
load digittest
load digittrain
load OutputWeight
clear batchdata batchtargets batchdigittargets
clear testbatchdata testbatchtargets batchdigittargetstest

N2 = length(targetstest);
%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases]; %(784+1*500)

digitdatatest = [digitdatatest ones(N2,1)];

w1probs = 1./(1 + exp(-digitdatatest*w1)); 
H = w1probs';  %700       10000
% size(H)

TV.T = targetstest'; %1 10000

clear digitdata targets digittargets 
clear digitdatatest targetstest digittargetstest 
clear mnistvhclassify vishid hidrecbiases visbiases

TY=(H' * OutputWeight)';                       %   TY: the actual output of the testing data
[TI test_output]=max(TY',[],2);%返回最大的预测标签的位置
[TI1 test_target]=max(TV.T',[],2);%返回最大的预测标签的位置

%%%%%%%%%% Calculate testing classification accuracy
test_accuracy = sum(test_output==test_target) / length(test_target)

t=toc

% train_accuracy =
% 
%     0.9620
% 
% 
% test_accuracy =
% 
%     0.9621
% 
% 
% t =
% 
%   897.3201




