function seq = load_video_info_OTB(video_path)

ground_truth = dlmread([video_path '\groundtruth_rect.txt']);
seq.ground_truth = ground_truth;        % ����groundtruth
seq.init_rect = ground_truth(1,:);      % ��ʼ������Ϊgroundtruth��һ�У�[x y w h]

target_sz = [ground_truth(1,4), ground_truth(1,3)];
seq.target_sz = target_sz;
seq.pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);

seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

img_path = [video_path '\img\'];

img_files_struct = dir(fullfile(img_path, '*.jpg'));
% img_files = img_files(4:end);
img_files = {img_files_struct.name};

% img_files = [img_path img_files];
% if exist([img_path num2str(1, '%04i.png')], 'file'),
%     img_files = num2str((1:seq.len)', [img_path '%04i.png']);
% elseif exist([img_path num2str(1, '%04i.jpg')], 'file'),
%     img_files = num2str((1:seq.len)', [img_path '%04i.jpg']);
% elseif exist([img_path num2str(1, '%04i.bmp')], 'file'),
%     img_files = num2str((1:seq.len)', [img_path '%04i.bmp']);
% else
%     error('No image files to load.')
% end
seq.s_frames = img_files(1,:);
% seq.s_frames = cellstr(img_files_struct);
seq.video_path = img_path;
for i = 1 : length(seq.s_frames)
    seq.s_frames{i} = [img_path seq.s_frames{i}];         % ÿһ֡������������·��
end
end