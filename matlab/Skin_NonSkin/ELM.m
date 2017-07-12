clc;
clear all;
close all;
tic;
load skintraindata 
% traintargets traintargetsbinary traindata 
load skintestdata 
% testtargets testtargetsbinary testdata 

load batchskintraindata 
% batchdata batchtargets
load batchskintestdata 
% testbatchdata testbatchtargets

NumberofHiddenNeurons = 300;
ActivationFunction = 'sig';

T = traintargetsbinary'; %10 60000
P = traindata'; %784 60000
TV.T = testtargetsbinary'; %10 10000
TV.P = testdata'; %784 10000

%========================================================================%
%==================================训练过程===============================%
%========================================================================%
NumberofTrainingData=size(P,2);   %列数
NumberofTestingData=size(TV.P,2); %列数
NumberofInputNeurons=size(P,1);   %行数
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;     %随机初始化输入层权重w
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);     %随机初始化隐层偏置b
tempH=InputWeight*P;        %P输入特征数据   （W.X）
clear P;                                            %   Release input of training data
%扩展的偏差矩阵纬度和H匹配
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here
end
clear tempH;
size(H) %700       60000
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
clear H;
[I train_output]=max(Y',[],2);%返回最大的预测标签的位置
[I1 train_target]=max(T',[],2);%返回最大的预测标签的位置

%%%%%%%%%% Calculate training classification accuracy
train_accuracy = sum(train_output==train_target) / length(train_target)

%========================================================================%
%==================================测试过程===============================%
%========================================================================%
%%%%%%%%%%% Calculate the output of testing input
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        %%%%%%%% More activation functions can be added here
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
[TI test_output]=max(TY',[],2);%返回最大的预测标签的位置
[TI1 test_target]=max(TV.T',[],2);%返回最大的预测标签的位置

%%%%%%%%%% Calculate testing classification accuracy
test_accuracy = sum(test_output==test_target) / length(test_target)

t = toc




















