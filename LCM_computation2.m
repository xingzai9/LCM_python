% 经典论文"A Local Contrast Method for Small Infrared Target Detection"中的Algorithm 1
% 输入patch的尺寸必须是3的倍数，考虑论文定义u的大小只能是3x3,5x5,7x7,9x9，patch的尺寸对应9x9,15x15,21x21,27x27
function  C_n  = LCM_computation2(patch_LCM_in)

[row,col] = size(patch_LCM_in);       % 对patch而言，行=列
patch_LCM_in = double(patch_LCM_in);  % 改数据类型
% 分为3x3个cells，无论patch的尺寸，都是3*3个cell
cell_size = row/3;
% 计算中心cell的最大值 是一个标量
% 方法一
% m_0 = mean(mean(cell_size+1:cell_size*2, cell_size+1:cell_size*2 ));
% L_n = max (max( patch_LCM_in( cell_size+1:cell_size*2, cell_size+1:cell_size*2 ) ) ); %选中
% L_n_2 = L_n*m_0;
% 方法二
L_n = max (max( patch_LCM_in( cell_size+1:cell_size*2, cell_size+1:cell_size*2 ) ) ); %选中patch中心区域，求最大值
L_n_2 = L_n^2;
% 计算周边cell的均值,周边共3^2-1个cell,编号如下：
% 1 2 3
% 4 0 5
% 6 7 8
m_1 = mean( mean( patch_LCM_in( 1:cell_size,                1:cell_size ) ));
m_2 = mean( mean( patch_LCM_in( 1:cell_size,                cell_size+1:cell_size*2 ) ));
m_3 = mean( mean( patch_LCM_in( 1:cell_size,                cell_size*2+1:cell_size*3 ) ));
m_4 = mean( mean( patch_LCM_in( cell_size+1:cell_size*2,    1:cell_size ) ));
m_5 = mean( mean( patch_LCM_in( cell_size+1:cell_size*2,    cell_size*2+1:cell_size*3 ) ));
m_6 = mean( mean( patch_LCM_in( cell_size*2+1:cell_size*3,  1:cell_size ) ));
m_7 = mean( mean( patch_LCM_in( cell_size*2+1:cell_size*3,  cell_size+1:cell_size*2 ) ));
m_8 = mean( mean( patch_LCM_in( cell_size*2+1:cell_size*3,  cell_size*2+1:cell_size*3 ) ));
% 计算C_n
m_cell = [L_n_2/m_1; L_n_2/m_2; L_n_2/m_3; L_n_2/m_4; L_n_2/m_5; L_n_2/m_6; L_n_2/m_7; L_n_2/m_8];
C_n = min(m_cell);
% Replace the value of the central pixel with the Cn
end
