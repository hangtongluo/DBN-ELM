clc;
clear all;
close all;

load skintestdata
load skintraindata
load dbnelmclonelayer
load dbnelmcltwolayer
load dbnelmclthreelayer
train_output = train_output -1;
test_output = test_output -1;
train = [train_output,yhatTrain1,yhatTrain2];
test = [test_output,yhatTest1,yhatTest2];
s1 = size(train);
s2 = size(test);
train_output = [];
test_output = [];

for i = 1:s1(1)
    if length(unique(train(i,:))) == 1
        train_output(i) = train(i,1);
    elseif length(unique(train(i,:))) == 3
        train_output(i) = train(i,1);
    else
        if train(i,1) == train(i,2)
            train_output(i) = train(i,1);
        elseif train(i,1) == train(i,3)
            train_output(i) = train(i,1);
        elseif train(i,2) == train(i,3)
            train_output(i) = train(i,2);
        end
    end
end

digittargets = traintargets;
train_accuracy_end = sum(reshape(train_output,s1(1),1) == digittargets) / length(digittargets)

for i = 1:s2(1)
    if length(unique(test(i,:))) == 1
        test_output(i) = test(i,1);
    elseif length(unique(test(i,:))) == 3
        test_output(i) = test(i,1);
    else
        if test(i,1) == test(i,2)
            test_output(i) = test(i,1);
        elseif test(i,1) == test(i,3)
            test_output(i) = test(i,1);
        elseif test(i,2) == test(i,3)
            test_output(i) = test(i,2);
        end
    end
end

digittargetstest = testtargets;
test_accuracy_end = sum(reshape(test_output,s2(1),1) == digittargetstest) / length(digittargetstest)




