% One pass evaluation for UAV123 dataset
% Fuling Lin, 20190101

function CPCF_Demo_all_seq(save_dir)  %  xxx��tracker�ļ�д; Ĭ��������'.\Test_yymmdd_hhmm\'
close all;
clc;

%% **Need to change**
where_is_your_groundtruth_folder = 'D:\Tracking\UAV123_10fps\anno';         % �����������ݼ�groundtruth�ļ���·��
where_is_your_UAV123_database_folder = 'D:\Tracking\UAV123_10fps\data_seq'; % �����������ݼ�ͼƬ���е�·��
tpye_of_assessment = 'UAV123_10fps';                                   % Ҫ���Ե����ͣ�'UAV123_10fps', 'UAV123', 'UAV123_20L'
tracker_name = 'CPCF_10fps';                                                 % Ҫ���Ե�tracker����

% where_is_your_groundtruth_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\anno\UAV20L';         % �����������ݼ�groundtruth�ļ���·��
% where_is_your_UAV123_database_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\data_seq\UAV123'; % �����������ݼ�ͼƬ���е�·��
% tpye_of_assessment = 'UAV123_20L';                                   % Ҫ���Ե����ͣ�'UAV123_10fps', 'UAV123', 'UAV123_20L'
% tracker_name = 'CPCF_20L';                                                 % Ҫ���Ե�tracker����

% where_is_your_groundtruth_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\anno\UAV123';         % �����������ݼ�groundtruth�ļ���·��
% where_is_your_UAV123_database_folder = 'D:\van\UAV123yang\Dataset_UAV123\UAV123\data_seq\UAV123'; % �����������ݼ�ͼƬ���е�·��
% tpye_of_assessment = 'UAV123';                                   % Ҫ���Ե����ͣ�'UAV123_10fps', 'UAV123', 'UAV123_20L'
% tracker_name = 'CPCF_30fps';   


setup_paths();                                                         %���·��


%% Read all video names using grouthtruth.txt
ground_truth_folder = where_is_your_groundtruth_folder;
dir_output = dir(fullfile(ground_truth_folder, '\*.txt'));             % ��ȡ���ļ����µ����е�txt�ļ�
contents = {dir_output.name}';
all_video_name = {};
for k = 1:numel(contents)
    name1 = contents{k}(1:end-4);                                       % ȥ����׺ .txt
    all_video_name{end+1,1} = name1;                                    % �����������ݼ�����
end
dataset_num = length(all_video_name);                                  % ��groundtruth���ļ����õ����ݼ�����
type = tpye_of_assessment;

set = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ];
    
    gamma = 0.8;
    gamma_str = num2str(gamma);
    
    lr = 0.0192;
    lr_str = num2str(lr);
        for dataset_count = 1 : dataset_num
            video_name = all_video_name{dataset_count}                          % ��ȡ���ݼ�����
            database_folder = where_is_your_UAV123_database_folder;            % �����������ݼ�ͼƬ���е�·��
            seq = load_video_info_UAV123(video_name, database_folder, ground_truth_folder, type); % ����������Ϣ
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

            
            % main function,ִ���㷨������������Ϊ�ṹ�壬���ٰ���type,res,fps������Ա
            result  =  run_CPCF(seq,0,0,gamma,lr);
           
           
            % save results
            results = cell(1,1);                                               % results�ǰ���һ���ṹ���Ԫ�����ṹ�����type,res,fps,len,annoBegin,startFrame������Ա
            results{1} = result;
            results{1}.len = seq.len;
            results{1}.startFrame = seq.st_frame;
            results{1}.annoBegin = seq.st_frame;
            
            
            fprintf('%s fps: %f \n', seq.name, results{1}.fps);
            
            % save results to specified folder
            if nargin < 1
                save_dir = '.\Test_for_fps\';              % ��������Ľ����ָ���ļ���
            end
            save_res_dir = [save_dir, tracker_name,'_gamma@', gamma_str ,'_lr@', lr_str '\'];                   % ����ͼƬ��·������KCFΪ����Ч���磺'.\all_trk_results\KCF_results\res_picture\'
            save_pic_dir = [save_res_dir,   'res_picture\'];                     % ����ͼƬ��·������KCFΪ����Ч���磺'.\all_trk_results\KCF_results\res_picture\'
            if ~exist(save_pic_dir, 'dir')
                mkdir(save_res_dir);
                mkdir(save_pic_dir);
            end
        save([save_res_dir, video_name, '_', tracker_name,'_gamma@', gamma_str,'_lr@', lr_str '.mat'], 'results');% ���ض����Ʊ������ݽ������KCF��bike1�Ľ��Ϊ����Ч���磺'.\all_trk_results\KCF_results\bike1_KCF.mat'
            
            % plot precision figure
            %
            show_visualization = 0;                                            % ��ʾͼƬ��precision_plot�����
            precision_plot_save(results{1}.res, seq.ground_truth, seq.video_name, save_pic_dir, show_visualization);
            
            close all;
%         end
        end


end
 