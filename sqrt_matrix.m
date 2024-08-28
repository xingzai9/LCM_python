% 用于计算矩阵方差，输入矩阵和均值，输出方差
function  sqrt_C_hat  = sqrt_matrix( C_hat,mean_C_hat )
C_hat = double(C_hat);
[row,col] = size(C_hat);
Num = row*col;
sum = 0;
for i=1:row
    for j=1:col
       sum = sum + (C_hat(i,j)- mean_C_hat)^2;
    end
end
sqrt_C_hat = sum/(Num-1);


end
