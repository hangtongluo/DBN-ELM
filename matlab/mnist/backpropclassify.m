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

% This program fine-tunes an autoencoder with backpropagation.
% Weights of the autoencoder are going to be saved in mnist_weights.mat
% and trainig and test reconstruction errors in mnist_error.mat
% You can also set maxepoch, default value is 200 as in our paper.

%maxepoch=200;
maxepoch=50;
% maxepoch=20;
% maxepoch=1;
fprintf(1,'\nTraining discriminative model on MNIST by minimizing cross entropy error. \n');
fprintf(1,'60 batches of 1000 cases each. \n');

load mnistvhclassify
load mnisthpclassify
load mnisthp2classify

makebatches;
[numcases numdims numbatches]=size(batchdata);
N=numcases;

%%%% PREINITIALIZE WEIGHTS OF THE DISCRIMINATIVE MODEL%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1=[vishid; hidrecbiases]; %(784+1*500)
w2=[hidpen; penrecbiases]; %(500+1*500)
w3=[hidpen2; penrecbiases2];%(500+1*2000)
%初始化数字标签列为（2001*10）
w_class = 0.1*randn(size(w3,2)+1,10);   %数字标签列（）


%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w_class,1)-1;
l5=10;
test_err=[];
train_err=[];

for epoch = 1:maxepoch
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    train_target = [];
    train_output = [];  
    test_target = [];
    test_output = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    [numcases numdims numbatches]=size(batchdata);%100 784 600
    N=numcases;
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)];
        target = [batchtargets(:,:,batch)];%增加标签列
        data = [data ones(N,1)];  %(100*785)
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];%(100*500+1)
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];%(100*500+1)
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];%(100*2000+1)
        
        %========================softmax==========================================
        targetout = exp(w3probs*w_class); %计算标签输出（100*10）=(((100*2001)*(2001*10)))
        targetout = targetout./repmat(sum(targetout,2),1,10);%约束到0-1之间
        
        %I表示每行的最大值，J表示每行的最大值的位置，find(J==J1)表示相同值得位置向量
        [I J]=max(targetout,[],2);%返回最大的预测标签的位置
        [I1 J1]=max(target,[],2);%返回最大的预测标签的位置
        counter=counter+length(find(J==J1));    %？？？？？？？？？？？？？？？？？？？
        err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))); %还原
        train_target = [train_target; J1];
        train_output = [train_output; J];
    end
    train_err(epoch)=(numcases*numbatches-counter);
    train_crerr(epoch)=err_cr/numbatches;
%     train_target = train_target - 1;
%     train_output = train_output - 1;
    %%%%%%%%%%%%%% END OF COMPUTING TRAINING MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%% COMPUTE TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    err=0;
    err_cr=0;
    counter=0;
    [testnumcases testnumdims testnumbatches]=size(testbatchdata);
    N=testnumcases;
    for batch = 1:testnumbatches
        data = [testbatchdata(:,:,batch)];
        target = [testbatchtargets(:,:,batch)];
        data = [data ones(N,1)];
        w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
        w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
        w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs ones(N,1)];
        targetout = exp(w3probs*w_class);
        targetout = targetout./repmat(sum(targetout,2),1,10);
        
        [I J]=max(targetout,[],2);
        [I1 J1]=max(target,[],2);
        counter=counter+length(find(J==J1));
        err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
        test_target = [test_target; J1];
        test_output = [test_output; J];
    end
    test_err(epoch)=(testnumcases*testnumbatches-counter);
    test_crerr(epoch)=err_cr/testnumbatches;
%     test_target = test_target -1;
%     test_output = test_output - 1;
    
    fprintf(1,'Before epoch %d Train # misclassified: %d (from %d). Test # misclassified: %d (from %d) \t \t \n',...
        epoch,train_err(epoch),numcases*numbatches,test_err(epoch),testnumcases*testnumbatches);
    
    %%%%%%%%%%%%%% END OF COMPUTING TEST MISCLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    tt=0;
    for batch = 1:numbatches/10
        fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%%%% COMBINE 10 MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt=tt+1;
        data=[];
        targets=[];
        for kk=1:10
            data=[data
                batchdata(:,:,(tt-1)*10+kk)];
            targets=[targets
                batchtargets(:,:,(tt-1)*10+kk)];
        end
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3;
        
        if epoch<6  % First update top-level weights holding other weights fixed.
            N = size(data,1);
            XX = [data ones(N,1)];
            w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
            w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
            w3probs = 1./(1 + exp(-w2probs*w3)); %w3probs = [w3probs  ones(N,1)];
            
            VV = [w_class(:)']';
            Dim = [l4; l5];
            [X, fX] = minimize(VV,'CG_CLASSIFY_INIT',max_iter,Dim,w3probs,targets);
            w_class = reshape(X,l4+1,l5);
            
        else
            VV = [w1(:)' w2(:)' w3(:)' w_class(:)']';
            Dim = [l1; l2; l3; l4; l5];
            [X, fX] = minimize(VV,'CG_CLASSIFY',max_iter,Dim,data,targets);
            
            w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
            xxx = (l1+1)*l2;
            w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
            xxx = xxx+(l2+1)*l3;
            w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
            xxx = xxx+(l3+1)*l4;
            w_class = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
            
        end
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    
    save mnistclassify_weights w1 w2 w3 w_class
    save mnistclassify_error test_err test_crerr train_err train_crerr;
    save train_output test_output
end



