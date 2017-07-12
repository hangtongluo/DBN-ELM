clc;
clear all;
close all;

%画随机初始化和迭代次数增多准确率变化
load mlelmdata
load dbnelmplotresult

index = [2,4,6,8,10];
number = [0,10,20,30,40,50];
% test_accuracy
% train_accuracy
% test_accuracy_num(index,1)
% train_accuracy_num(index,1)

test = [test_accuracy;test_accuracy_num(index,1)]
train = [train_accuracy;train_accuracy_num(index,1)]

plot(number, train, '--r.', number, test, '--b.')
legend('训练准确率曲线', '测试准确率曲线', 4)
xlabel('The number of iterations')
ylabel('accuracy')
axis([0,50,0.8,1]) % axis([xmin,xmax,ymin,ymax])，用这个语句可以对x，y轴的上限与下限绘制范围一起做控制




































































