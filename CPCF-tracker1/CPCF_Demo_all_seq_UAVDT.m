function run_all_trackers_UAVDT(save_dir)                                        % û�����룬��Ĭ�ϱ����ڵ�ǰ�ļ����µ�\all_trk_results\
where_is_your_groundtruth_folder = 'D:\Tracking\UAVDT\anno';         % �����������ݼ�groundtruth�ļ���·��
where_is_your_UAVDT_database_folder = 'D:\Tracking\UAVDT\data_seq'; % �����������ݼ�ͼƬ���е�·��
type_of_assessment = 'UAVDT';                                   % Ҫ���Ե����ͣ�'UAV123_10fps', 'UAV123', 'UAV123_20L'
tracker_name = 'CPCF_UAVDT';  

%%
setup_paths();                                                         %���·��

%% Read all video names using grouthtruth.txt
type = type_of_assessment;
ground_truth_folder = where_is_your_groundtruth_folder;
dir_output = dir(fullfile(ground_truth_folder, '\*.txt'));             % ��ȡ���ļ����µ����е�txt�ļ�
contents = {dir_output.name}';
all_video_name = {};
for k = 1:numel(contents)
    name = contents{k}(1:end-7);                                       % ȥ����׺ .txt
    all_video_name{end+1,1} = name;                                    % �����������ݼ�����
end
dataset_num = length(all_video_name);                                  % ��groundtruth���ļ����õ����ݼ�����

% main_folder = pwd;                                                     % ��ȡ��ǰ·��
% all_trackers_dir = '.\tracker_set\';                                   % ��������tracker���ļ���
% run_trackers_info = trackers_info();                                   % ��ȡ����tracker�ĺ�����Ϣ�����������run_xxx(seq, res_path, bSaveImage))������ʽ
% tracker_name_set = fieldnames(run_trackers_info);                      % ��ȡtracker_set�ĳ�Ա��
% tracker_num = length(tracker_name_set);                                % ��ȡtracker_set�������
% cd(all_trackers_dir);                                                  % �����������tracker���ļ���
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
        video_name = all_video_name{dataset_count}                    % ��ȡ���ݼ�����
        database_folder = where_is_your_UAVDT_database_folder;
        
        %             seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type); % ����������Ϣ
        seq = load_video_info_UAVDT(database_folder, ground_truth_folder, video_name);% ����������Ϣ
        res_path = '.\res\';
        
        assignin('base','subS',seq);                                   % ��seqд�빤���ռ䣬����ΪsubS
       
            
        result  =  run_CPCF(seq,0,0,gamma,lr);

        % UAVDTֻ���� type, res, fps, len �ĸ��ֶ�
        % save results
        results = cell(1,1);                                           % results�ǰ���һ���ṹ���Ԫ�����ṹ�����type,res,fps,len,annoBegin,startFrame������Ա
        results{1} = result;
        results{1}.len = seq.len;
        fprintf('%d %s----fps: %f\n', dataset_count, video_name, results{1}.fps);
        
        % save results to specified folder
        if nargin < 1
            save_dir = '.\Test_for_fps\';              % ��������Ľ����ָ���ļ���
        end
        save_res_dir = [save_dir, tracker_name,'_gamma@', gamma_str ,'_lr@', lr_str '\'];          % �������ݽ����·������KCFΪ����Ч���磺'.\all_trk_results\KCF_results\'
        save_pic_dir = [save_res_dir, 'res_picture\'];                 % ����ͼƬ��·������KCFΪ����Ч���磺'.\all_trk_results\KCF_results\res_picture\'
        if ~exist(save_pic_dir, 'dir')
            mkdir(save_res_dir);
            mkdir(save_pic_dir);
        end
        save([save_res_dir, video_name, '_', tracker_name,'_gamma@', gamma_str,'_lr@', lr_str '.mat'], 'results');% ���ض����Ʊ������ݽ������KCF��bike1�Ľ��Ϊ����Ч���磺'.\all_trk_results\KCF_results\bike1_KCF.mat'
        
        % plot precision figure
        show_visualization = 0;                                        % ��ʾͼƬ��precision_plot�����
        precision_plot_save(results{1}.res, seq.ground_truth, video_name, save_pic_dir, show_visualization);
        close all;
    end

%     cd ..;                                                             % �ص���������tracker���ļ�����
%     rmpath(genpath(tracker_name));                                     % �Ƴ��ļ����Լ��������ļ��е�·��
