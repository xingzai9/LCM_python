% 根据阈值判断，输出二值探测结果和统计小目标在mask中占据的像元数
% 输入：对比度图C_hat，阈值threshold，空区宽度max_margin，待检测输入图像
function  [I_out,target_pixel_num]  = target_detection(C_hat,threshold,max_margin,I_in)
% 用阈值生成mask
[row,col] = size(C_hat);
mask = zeros(row,col);
target_pixel_num = 0;   % 统计小目标在mask中占据的像元数
for i = 1:row
    for j = 1:col
        if C_hat(i,j)>threshold
            mask(i, j)=1;
            target_pixel_num = target_pixel_num+1;
        end
    end
end
clear row; clear col; clear i; clear j;

% 再把mask填入原图的中区域，四周各空max_margin,max_margin取值与MLCM_computation函数中计算过程有关,是最大27*27patch的一半
[row,col] = size(I_in);
I_out = zeros(row,col);
I_out(max_margin+1:row-max_margin,max_margin+1:col-max_margin) = mask;
clear row; clear col;
end
