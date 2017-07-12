clc;
clear all;
close all;


M = load('Skin_NonSkin.txt') ;

index = [50859,194198];
index1 = round(index(1) * 0.7);
index2 = round(index(2) * 0.7);

train_x1 = M(1:index1,1:3); 
target_train_x1 = M(1:index1,4);
binary_target_train_x1 = repmat([1,0],length(target_train_x1),1);
test_x1 = M(index1+1:index(1),1:3); 
target_test_x1 = M(index1+1:index(1),4);
binary_target_test_x1 = repmat([1,0],length(target_test_x1),1);

train_x2 = M(index(1):index2,1:3); 
target_train_x2 = M(index(1):index2,4);
binary_target_train_x2 = repmat([0,1],length(target_train_x2),1);
test_x2 = M(index2+1:index(2),1:3); 
target_test_x2 = M(index2+1:index(2),4);
binary_target_test_x2 = repmat([0,1],length(target_test_x2),1);

train_x = [train_x1;train_x2];
train_y = [target_train_x1;target_train_x2];
binary_train_y = [binary_target_train_x1;binary_target_train_x2];
test_x = [test_x1;test_x2];
test_y = [target_test_x1;target_test_x2];
binary_test_y = [binary_target_test_x1;binary_target_test_x2];


traindata = train_x(1:120600,:);
traintargetsbinary = binary_train_y(1:120600,:);
traintargets = train_y(1:120600);

%将数据进行分块
traindata = traindata/255;
totnum=size(traindata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);
rand('state',2017); 
randomorder=randperm(totnum);  %打乱顺序
numbatches=totnum/100;  %112000个序列/100
numdims  =  size(traindata,2);   %625=25*25
batchsize = 100;
batchdata = zeros(batchsize, numdims, numbatches); % 1000   625   1120
batchtargets = zeros(batchsize, 2, numbatches);  % 1000   40   1120
%把数据存储到三维的分批矩阵中
for b=1:numbatches
    batchdata(:,:,b) = traindata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets(:,:,b) = traintargetsbinary(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;

testdata = test_x(1:73500,:);
testtargetsbinary = binary_test_y(1:73500,:);
testtargets = test_y(1:73500);

testdata = testdata/255;
totnum=size(testdata,1);
fprintf(1, 'Size of the trsting dataset= %5d \n', totnum);
rand('state',2017); 
randomorder=randperm(totnum);  %打乱顺序
numbatches=totnum/100;  %48000个序列/1000
numdims  =  size(testdata,2);   %625=25*25
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches); % 10   625   400
testbatchtargets = zeros(batchsize, 2, numbatches);  % 10   40   400
%把数据存储到三维的分批矩阵中
for b=1:numbatches
    testbatchdata(:,:,b) = testdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    testbatchtargets(:,:,b) = testtargetsbinary(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;


dos('del skintraindata.mat');
dos('del skintestdata.mat');
dos('del batchskintraindata.mat');
dos('del batchskintestdata.mat');

save skintraindata traintargets traintargetsbinary traindata 
save skintestdata testtargets testtargetsbinary testdata 

save batchskintraindata batchdata batchtargets
save batchskintestdata testbatchdata testbatchtargets

%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));
% 

























