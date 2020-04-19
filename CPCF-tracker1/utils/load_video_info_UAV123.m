% Loads relevant information of UAV123 in the given path.
% Fuling Lin, 20190101

function seq = load_video_info_UAV123(video_name, database_folder, ground_truth_path, type)

    switch type
        case 'UAV123_10fps'
            seqs = configSeqs(database_folder, type);       % database_folder�ǰ������������ݼ����ļ���
        case 'UAV123'
            seqs = configSeqs(database_folder, type);       % type: UAV123_10fps, UAV123, UAV123_20L
        case 'UAV123_20L'
            seqs = configSeqs(database_folder, type);
    end
    
    i=1;
    while ~strcmpi(seqs{i}.name,video_name) % Ϊ�˻��configSeqs�ж���ѡ�����ݼ�������
            i=i+1;
    end
    
    seq.video_name = seqs{i}.name;          % ������ݼ����ƣ���video_name��ͬ
    seq.name = seqs{i}.name;
    seq.video_path = seqs{i}.path;          % ���ݼ�����·��������ͼƬ���е��ļ���
    seq.st_frame = seqs{i}.startFrame;      % ��ʼ֡��
    seq.en_frame = seqs{i}.endFrame;        % ����֡��
    seq.len = seq.en_frame-seq.st_frame+1;  % ���г���
    
    ground_truth = dlmread([ground_truth_path '\' seq.video_name '.txt']);
    seq.ground_truth = ground_truth;        % ����groundtruth
    
    seq.init_rect = ground_truth(1,:);      % ��ʼ������Ϊgroundtruth��һ�У�[x y w h]    
    target_sz = [ground_truth(1,4), ground_truth(1,3)];
    seq.target_sz = target_sz;
	seq.pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
    
    img_path = seq.video_path;
    img_files_struct = dir(fullfile(img_path, '*.jpg'));
    img_files = {img_files_struct.name};                      % ������ͼƬ���Ʊ���Ϊcell����\
    seq.img_files = img_files;
    seq.s_frames = img_files(1, seq.st_frame : seq.en_frame); % ȡ��configSeq�������õ���Ч֡����
    for i = 1 : length(seq.s_frames)
        seq.s_frames{i} = [img_path seq.s_frames{i}];         % ÿһ֡������������·��
    end
    