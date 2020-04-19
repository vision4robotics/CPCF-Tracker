% One pass evaluation for UAV123 dataset
% Fuling Lin, 20190101

function CPCF_Demo_all_seq(save_dir)  %  xxx是tracker的简写; 默认输入是'.\Test_yymmdd_hhmm\'
close all;
clc;

%% **Need to change**
where_is_your_groundtruth_folder = 'D:\Tracking\UAV123_10fps\anno';         % 包含所有数据集groundtruth文件的路径
where_is_your_UAV123_database_folder = 'D:\Tracking\UAV123_10fps\data_seq'; % 包含所有数据集图片序列的路径
tpye_of_assessment = 'UAV123_10fps';                                   % 要测试的类型，'UAV123_10fps', 'UAV123', 'UAV123_20L'
tracker_name = 'CPCF_10fps';                                                 % 要测试的tracker名称

% where_is_your_groundtruth_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\anno\UAV20L';         % 包含所有数据集groundtruth文件的路径
% where_is_your_UAV123_database_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\data_seq\UAV123'; % 包含所有数据集图片序列的路径
% tpye_of_assessment = 'UAV123_20L';                                   % 要测试的类型，'UAV123_10fps', 'UAV123', 'UAV123_20L'
% tracker_name = 'CPCF_20L';                                                 % 要测试的tracker名称

% where_is_your_groundtruth_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\anno\UAV123';         % 包含所有数据集groundtruth文件的路径
% where_is_your_UAV123_database_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\data_seq\UAV123'; % 包含所有数据集图片序列的路径
% tpye_of_assessment = 'UAV123';                                   % 要测试的类型，'UAV123_10fps', 'UAV123', 'UAV123_20L'
% tracker_name = 'CPCF_30fps';   


setup_paths();                                                         %添加路径


%% Read all video names using grouthtruth.txt
ground_truth_folder = where_is_your_groundtruth_folder;
dir_output = dir(fullfile(ground_truth_folder, '\*.txt'));             % 获取该文件夹下的所有的txt文件
contents = {dir_output.name}';
all_video_name = {};
for k = 1:numel(contents)
    name1 = contents{k}(1:end-4);                                       % 去掉后缀 .txt
    all_video_name{end+1,1} = name1;                                    % 保存所有数据集名称
end
dataset_num = length(all_video_name);                                  % 从groundtruth总文件数得到数据集总数
type = tpye_of_assessment;

set = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ];
    
    gamma = 0.8;
    gamma_str = num2str(gamma);
    
    lr = 0.0192;
    lr_str = num2str(lr);
        for dataset_count = 1 : dataset_num
            video_name = all_video_name{dataset_count}                          % 读取数据集名称
            database_folder = where_is_your_UAV123_database_folder;            % 包含所有数据集图片序列的路径
            seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type); % 加载序列信息
            res_path = '.\res\';

            
                
%             seq.len = seq.en_frame - seq.st_frame + 1;
%             seq.s_frames = cell(seq.len,1);
%             nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
%             for i=1:seq.len
%                 image_no = seq.st_frame + (i-1);
%                 id = sprintf(nz,image_no);
%                 seq.s_frames{i} = strcat(seq.path,id,'.',seq.ext);
%             end
%             

            
            % main function,执行算法主函数，返回为结构体，至少包含type,res,fps三个成员
            result  =  run_CPCF(seq,0,0,gamma,lr);
           
           
            % save results
            results = cell(1,1);                                               % results是包含一个结构体的元胞，结构体包括type,res,fps,len,annoBegin,startFrame六个成员
            results{1} = result;
            results{1}.len = seq.len;
            results{1}.startFrame = seq.st_frame;
            results{1}.annoBegin = seq.st_frame;
            
            
            fprintf('%s fps: %f \n', seq.name, results{1}.fps);
            
            % save results to specified folder
            if nargin < 1
                save_dir = '.\Test_for_fps\';              % 保存跑完的结果到指定文件夹
            end
            save_res_dir = [save_dir, tracker_name,'_gamma@', gamma_str ,'_lr@', lr_str '\'];                   % 保存图片的路径，以KCF为例，效果如：'.\all_trk_results\KCF_results\res_picture\'
            save_pic_dir = [save_res_dir,   'res_picture\'];                     % 保存图片的路径，以KCF为例，效果如：'.\all_trk_results\KCF_results\res_picture\'
            if ~exist(save_pic_dir, 'dir')
                mkdir(save_res_dir);
                mkdir(save_pic_dir);
            end
        save([save_res_dir, video_name, '_', tracker_name,'_gamma@', gamma_str,'_lr@', lr_str '.mat'], 'results');% 以特定名称保存数据结果，以KCF跑bike1的结果为例，效果如：'.\all_trk_results\KCF_results\bike1_KCF.mat'
            
            % plot precision figure
            %
            show_visualization = 0;                                            % 显示图片（precision_plot）结果
            precision_plot_save(results{1}.res, seq.ground_truth, seq.video_name, save_pic_dir, show_visualization);
            
            close all;
%         end
        end


end
 