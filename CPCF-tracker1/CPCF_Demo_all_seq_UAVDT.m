function run_all_trackers_UAVDT(save_dir)                                        % 没有输入，则默认保存在当前文件下下的\all_trk_results\
where_is_your_groundtruth_folder = 'D:\Tracking\UAVDT\anno';         % 包含所有数据集groundtruth文件的路径
where_is_your_UAVDT_database_folder = 'D:\Tracking\UAVDT\data_seq'; % 包含所有数据集图片序列的路径
type_of_assessment = 'UAVDT';                                   % 要测试的类型，'UAV123_10fps', 'UAV123', 'UAV123_20L'
tracker_name = 'CPCF_UAVDT';  

%%
setup_paths();                                                         %添加路径

%% Read all video names using grouthtruth.txt
type = type_of_assessment;
ground_truth_folder = where_is_your_groundtruth_folder;
dir_output = dir(fullfile(ground_truth_folder, '\*.txt'));             % 获取该文件夹下的所有的txt文件
contents = {dir_output.name}';
all_video_name = {};
for k = 1:numel(contents)
    name = contents{k}(1:end-7);                                       % 去掉后缀 .txt
    all_video_name{end+1,1} = name;                                    % 保存所有数据集名称
end
dataset_num = length(all_video_name);                                  % 从groundtruth总文件数得到数据集总数

% main_folder = pwd;                                                     % 获取当前路径
% all_trackers_dir = '.\tracker_set\';                                   % 包含所有tracker的文件夹
% run_trackers_info = trackers_info();                                   % 获取运行tracker的函数信息，函数最好是run_xxx(seq, res_path, bSaveImage))这种形式
% tracker_name_set = fieldnames(run_trackers_info);                      % 获取tracker_set的成员名
% tracker_num = length(tracker_name_set);                                % 获取tracker_set里的总数
% cd(all_trackers_dir);                                                  % 进入包含所有tracker的文件夹
%%
    
set = [
    0.015, 0.5;
    0.015,0;
    0.016,0.5;
    0.016,0;
    0.017,0.5; 
    0.017,0;
    0.018,0.5;
    0.018,0;
    0.019,0.5;
    0.019,0;
    0.020,0.5;
    0.020,0;
    0.021,0.5;
    0.021,0;
    0.022,0.5;
    0.022,0;
    0.023,0.5;
    0.023,0;
    0.024,0.5;
    0.024,0;
    ];

    gamma = 0.8;
    gamma_str = num2str(gamma);
    
    lr = 0.0192;
    lr_str = num2str(lr);

    for dataset_count = 1 : dataset_num
        video_name = all_video_name{dataset_count}                    % 读取数据集名称
        database_folder = where_is_your_UAVDT_database_folder;
        
        %             seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type); % 加载序列信息
        seq = load_video_info_UAVDT(database_folder, ground_truth_folder, video_name);% 加载序列信息
        res_path = '.\res\';
        
        assignin('base','subS',seq);                                   % 将seq写入工作空间，命名为subS
       
            
        result  =  run_CPCF(seq,0,0,gamma,lr);

        % UAVDT只包含 type, res, fps, len 四个字段
        % save results
        results = cell(1,1);                                           % results是包含一个结构体的元胞，结构体包括type,res,fps,len,annoBegin,startFrame六个成员
        results{1} = result;
        results{1}.len = seq.len;
        fprintf('%d %s----fps: %f\n', dataset_count, video_name, results{1}.fps);
        
        % save results to specified folder
        if nargin < 1
            save_dir = '.\Test_for_fps\';              % 保存跑完的结果到指定文件夹
        end
        save_res_dir = [save_dir, tracker_name,'_gamma@', gamma_str ,'_lr@', lr_str '\'];          % 保存数据结果的路径，以KCF为例，效果如：'.\all_trk_results\KCF_results\'
        save_pic_dir = [save_res_dir, 'res_picture\'];                 % 保存图片的路径，以KCF为例，效果如：'.\all_trk_results\KCF_results\res_picture\'
        if ~exist(save_pic_dir, 'dir')
            mkdir(save_res_dir);
            mkdir(save_pic_dir);
        end
        save([save_res_dir, video_name, '_', tracker_name,'_gamma@', gamma_str,'_lr@', lr_str '.mat'], 'results');% 以特定名称保存数据结果，以KCF跑bike1的结果为例，效果如：'.\all_trk_results\KCF_results\bike1_KCF.mat'
        
        % plot precision figure
        show_visualization = 0;                                        % 显示图片（precision_plot）结果
        precision_plot_save(results{1}.res, seq.ground_truth, video_name, save_pic_dir, show_visualization);
        close all;
    end

%     cd ..;                                                             % 回到包含所有tracker的文件夹中
%     rmpath(genpath(tracker_name));                                     % 移除文件夹以及所有子文件夹的路径
