clc;
clear all;
close all;
train_accuracy_num = [];
test_accuracy_num = [];
num = 0;
train_accuracy_num_two = [];
test_accuracy_num_two = [];
for i = 100:100:600
    for j = 100:100:600
        for k = 100:100:600
            [i,j,k]
            num = num + 1;
            load digittest
            load digittrain
            clear batchdata batchtargets batchdigittargets
            clear testbatchdata testbatchtargets batchdigittargetstest
            numhid=i; numpen=j; numpen2=k;
            [N1,N] = size(digitdata);
            vishid = randn(N,numhid);
            hidpen = randn(numhid,numpen);
            hidpen2 = randn(numpen,numpen2);
            hidrecbiases = randn(1,numhid);
            penrecbiases = randn(1,numpen);
            penrecbiases2 = randn(1,numpen2);
            
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
%             clear digitdata targets digittargets
%             clear digitdatatest targetstest digittargetstest
            
            OutputWeight=pinv(H') * T';                        % implementation without regularization factor //refer to 2006 Neurocomputing paper
            save OutputWeight
            %%%%%%%%%%% Calculate the training accuracy
            Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
            clear H;
            [I train_output]=max(Y',[],2);%返回最大的预测标签的位置
            [I1 train_target]=max(T',[],2);%返回最大的预测标签的位置
            
            %%%%%%%%%% Calculate training classification accuracy
            train_accuracy = sum(train_output==train_target) / length(train_target)
            train_accuracy_num(num) = train_accuracy;
            train_accuracy_num_two(i,j,k) = train_accuracy;
            %=======================================================================%
            %===========================测试过程=====================================%
            %=======================================================================%
          
            load OutputWeight
            
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
            
            
            TY=(H' * OutputWeight)';                       %   TY: the actual output of the testing data
            [TI test_output]=max(TY',[],2);%返回最大的预测标签的位置
            [TI1 test_target]=max(TV.T',[],2);%返回最大的预测标签的位置
            
            %%%%%%%%%% Calculate testing classification accuracy
            test_accuracy = sum(test_output==test_target) / length(test_target)
            test_accuracy_num(i) = test_accuracy;
            test_accuracy_num_two(i,j,k) = test_accuracy;
        end
    end
end
save mlelmplotresult train_accuracy_num test_accuracy_num train_accuracy_num_two test_accuracy_num_two

plot(train_accuracy_num, '--r.')
plot(number, test_accuracy_num, '--b.')
legend('训练准确率曲线', '测试准确率曲线', 2)
xlabel('神经元规模和层数')
ylabel('accuracy')





