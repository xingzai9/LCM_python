
cd 'D:\newdesktop\LCM'; % 改路径后即可直接运行！！！

% 读取测试图像
% I_read = imread("in.png");      %可以成功的图像
I_read = imread("unsuccess.png"); %测试不成功的图像
% I_read = imread("test.bmp");
I_read = imresize(I_read,[256 256],'nearest'); %表示把图像P重塑成指定大小，使用指定的插值方法

% 检测RGB，转为灰度图
if size(I_read, 3) == 3
    I_in = rgb2gray(I_read);	
else
    I_in = I_read;	
end
clear I_read;
% 显示输入图像，转为double型
% imshow(I_in);   % imtool(I_in);
I_in = double(I_in);

% 计算最终的Multiscale LCM，注意这张图小于原始图像，最大的patch为27*27，因此比原始图像小26
[C_hat,max_margin]  = MLCM_computation2(I_in);
% 计算均值、标准差
mean_C_hat = mean(C_hat(:));   % 矩阵均值
sqrt_C_hat = (sqrt_matrix(C_hat,mean_C_hat))^0.5; % 标准差
% 计算阈值
k_Th = 4;
threshold = mean_C_hat + k_Th*sqrt_C_hat;
% 根据阈值判断，输出二值探测结果和统计小目标在mask中占据的像元数
[I_out,target_pixel_num]  = target_detection(C_hat,threshold,max_margin,I_in);

% 显示输入和输出
figure()
subplot(1,2,1); imshow(I_in,[0,255]),impixelinfo; title('原图');
subplot(1,2,2); imshow(I_out),impixelinfo; title('二值化输出');
