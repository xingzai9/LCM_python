% 经典论文"A Local Contrast Method for Small Infrared Target Detection"中的Algorithm 2
% 输入patch的尺寸必须是3的倍数，考虑论文定义u的大小只能是3x3,5x5,7x7,9x9，patch的尺寸对应9x9,15x15,21x21,27x27
function [C_hat,max_margin]  = MLCM_computation2(I_MLCM_in)
I_MLCM_in = double(I_MLCM_in);
[row,col] = size(I_MLCM_in);       
scales = [9,15,21,27];   % patch的尺寸有9x9,15x15,21x21,27x27
l_max = size(scales);    % 对应论文lmax=[1,4],l_max(2)是4

% Compute Cl according to Algorithm 1
C_map_scales = zeros(row,col,l_max(2)); 
for i = 1:l_max(2)          % 对应不同尺度
    for j = 1:row-scales(i)+1  % 单一尺度下以patch为单位做遍历，j是行
        for k = 1:col-scales(i)+1                             %k是列

            temp_patch = I_MLCM_in(j:j+scales(i)-1, k:k+scales(i)-1);%从（1，1）（左上角那个点）开始大小为scales(i)×scales(i)作为输入
            %原代码每次滑动窗口后会覆盖上一次的值
            C_n  = LCM_computation2(temp_patch);    % 对patch执行Algorithm 1
            C_map_scales((2*j+scales(i)-1)/2, (2*k+scales(i)-1)/2,i) = C_n;
        end
    end
end
% 这部分计算，生成4张对比度图，其中尺度最大的对比度图有效像元数最小，每个方向减去(scales(4)-1)/2=13
max_margin = (scales(4)-1)/2;

% 输出4种尺度的对比图
figure()
[X,Y] = meshgrid(1:1:row,1:1:col);
subplot(2,2,1); mesh(X,Y,C_map_scales(:,:,1)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=3x3 Contrast Map');
subplot(2,2,2); mesh(X,Y,C_map_scales(:,:,2)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=5x5 Contrast Map');
subplot(2,2,3); mesh(X,Y,C_map_scales(:,:,3)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=7x7 Contrast Map');
subplot(2,2,4); mesh(X,Y,C_map_scales(:,:,4)); axis([0 row 0 col 0 255]);xlabel('row');ylabel('col');zlabel('value'); title('v=9x9 Contrast Map');

% 对4种尺度对比图的共同部分取最大值，作为输出
C_hat = zeros(row-scales(4)+1,col-scales(4)+1);
for i = 1:row-scales(4)+1
    for j = 1:row-scales(4)+1
        temp = [C_map_scales(i+max_margin,j+max_margin,1);...
                C_map_scales(i+max_margin,j+max_margin,2);...
                C_map_scales(i+max_margin,j+max_margin,3);...
                C_map_scales(i+max_margin,j+max_margin,4)];
        C_hat(i,j) = max(temp);
    end
end

end
