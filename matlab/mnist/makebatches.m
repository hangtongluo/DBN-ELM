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

digitdata=[];
targets=[];
digittargets = [];
load digit0; digitdata = [digitdata; D]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*0];
load digit1; digitdata = [digitdata; D]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*1];
load digit2; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*2];
load digit3; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*3];
load digit4; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*4];
load digit5; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*5];
load digit6; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*6];
load digit7; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*7];
load digit8; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*8];
load digit9; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digittargets = [digittargets; ones(size(D,1), 1)*9];
digitdata = digitdata/255;      %预处理到【0-1】范围内
%size(digitdata)        60000         784
%size(targets)           %60000          10
totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);  %打乱顺序

numbatches=totnum/100;  %60000个序列/100
numdims  =  size(digitdata,2);   %784=28*28
batchsize = 100;
%600个100*784的矩阵
batchdata = zeros(batchsize, numdims, numbatches); % 100   784   600
batchtargets = zeros(batchsize, 10, numbatches);  % 100   10   600
batchdigittargets = zeros(batchsize, numbatches);  %100 600
%把数据存储到三维的分批矩阵中
for b=1:numbatches
    batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchdigittargets(:,b) = digittargets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
save digittrain digitdata targets digittargets batchdata batchtargets batchdigittargets
clear digitdata targets digittargets;

digitdatatest=[];
targetstest=[];
digittargetstest = [];
load test0; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*0];
load test1; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*1];
load test2; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*2];
load test3; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*3];
load test4; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*4];
load test5; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*5];
load test6; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*6];
load test7; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*7];
load test8; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*8];
load test9; digitdatatest = [digitdatatest; D]; targetstest = [targetstest; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digittargetstest = [digittargetstest; ones(size(D,1), 1)*9];
digitdatatest = digitdatatest/255;   %预处理到【0-1】范围内
%size(digitdata)         10000         784
totnum=size(digitdatatest,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdatatest,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 10, numbatches);
batchdigittargetstest = zeros(batchsize, numbatches);
for b=1:numbatches
    testbatchdata(:,:,b) = digitdatatest(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    testbatchtargets(:,:,b) = targetstest(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchdigittargetstest(:,b) = digittargetstest(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
save digittest digitdatatest targetstest digittargetstest testbatchdata testbatchtargets batchdigittargetstest
clear digitdatatest targetstest;


%%% Reset random seeds
rand('state',sum(100*clock));
randn('state',sum(100*clock));



