% ������ֵ�жϣ������ֵ̽������ͳ��СĿ����mask��ռ�ݵ���Ԫ��
% ���룺�Աȶ�ͼC_hat����ֵthreshold���������max_margin�����������ͼ��
function  [I_out,target_pixel_num]  = target_detection(C_hat,threshold,max_margin,I_in)
% ����ֵ����mask
[row,col] = size(C_hat);
mask = zeros(row,col);
target_pixel_num = 0;   % ͳ��СĿ����mask��ռ�ݵ���Ԫ��
for i = 1:row
    for j = 1:col
        if C_hat(i,j)>threshold
            mask(i, j)=1;
            target_pixel_num = target_pixel_num+1;
        end
    end
end
clear row; clear col; clear i; clear j;

% �ٰ�mask����ԭͼ�����������ܸ���max_margin,max_marginȡֵ��MLCM_computation�����м�������й�,�����27*27patch��һ��
[row,col] = size(I_in);
I_out = zeros(row,col);
I_out(max_margin+1:row-max_margin,max_margin+1:col-max_margin) = mask;
clear row; clear col;
end
