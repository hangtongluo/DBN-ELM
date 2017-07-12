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

% This program reads raw MNIST files available at
% http://yann.lecun.com/exdb/mnist/
% and converts them to files in matlab format
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.

% This program was originally written by Yee Whye Teh

% Work with test files first
%fprintf(fid, format, variables) //fid=1 for standard output (the screen)
fprintf(1,'You first need to download files:\n train-images-idx3-ubyte.gz\n train-labels-idx1-ubyte.gz\n t10k-images-idx3-ubyte.gz\n t10k-labels-idx1-ubyte.gz\n from http://yann.lecun.com/exdb/mnist/\n and gunzip them \n');

%%
%===================去掉图片和数字说明前面的文字说明==============
%%
f = fopen('t10k-images.idx3-ubyte','r');  %打开成功返回文件句柄值大于0
%“fread”以二进制形式，从文件读出数据。
%[a,count]=fread(fid,size,precision) precision 和原文件一致
[a,count] = fread(f,4,'int32');    %打开相应句柄值对应的文件

g = fopen('t10k-labels.idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

%%
%======================转换数据=============================
%%
fprintf(1,'Starting to convert Test MNIST images (prints 10 dots) \n');
%创建10个矩阵用于写入ascii数据
n = 1000;
Df = cell(1,10);
for d=0:9,
    Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');  %
end;

for i=1:10,
    fprintf('.');
    rawimages = fread(f,28*28*n,'uchar');  %读取图像的大小28*28*n
    rawlabels = fread(g,n,'uchar');       %读取标签的大小n
    rawimages = reshape(rawimages,28*28,n);%转换成28*28行1000列
    %imshow(reshape(rawimages(:,100),28,28));
    %写入10个test中分开存储
    for j=1:n,
        fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));%对应标签和图像写入内存
        fprintf(Df{rawlabels(j)+1},'\n');
    end;
end;

fprintf(1,'\n');
for d=0:9,
    fclose(Df{d+1});
    D = load(['test' num2str(d) '.ascii'],'-ascii');
    fprintf('%5d Digits of class %d\n',size(D,1),d);   %显示转换的数据大小
    save(['test' num2str(d) '.mat'],'D','-mat');  %把数据D存储到此文件
end;


% Work with training files second

f = fopen('train-images.idx3-ubyte','r');
[a,count] = fread(f,4,'int32');

g = fopen('train-labels.idx1-ubyte','r');
[l,count] = fread(g,2,'int32');

fprintf(1,'Starting to convert Training MNIST images (prints 60 dots)\n');
n = 1000;

Df = cell(1,10);
for d=0:9,
    Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
end;

for i=1:60,
    fprintf('.');
    rawimages = fread(f,28*28*n,'uchar');
    rawlabels = fread(g,n,'uchar');
    rawimages = reshape(rawimages,28*28,n);
    
    for j=1:n,
        fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
        fprintf(Df{rawlabels(j)+1},'\n');
    end;
end;

fprintf(1,'\n');
for d=0:9,
    fclose(Df{d+1});
    D = load(['digit' num2str(d) '.ascii'],'-ascii');
    fprintf('%5d Digits of class %d\n',size(D,1),d);
    save(['digit' num2str(d) '.mat'],'D','-mat');
end;

dos('del *.ascii'); %调用系统命令，运行dos命令删除.ascii文件
%dos('rm *.ascii');



