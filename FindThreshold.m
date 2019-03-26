function[]  =FindThreshold()

% Split 10^5 features into batches of 10000, so that we can load positive
% and negative values of a feature in to RAM.
clear all;

% load('face_feat.mat')
% 
% face_feat_mean = zeros(size(f,1),1);
% 
% for i = 1:size(f,1)
%            face_feat_mean(i) = mean(f(i,:));
% end
% 
% save('face_feat_mean.mat','face_feat_mean','-mat','-v7.3');

% load('nonface_feat3.mat')
% 
% nonface_feat_mean = zeros(size(f_non,1),1);
% 
% for i = 1:size(f_non,1)
%            nonface_feat_mean(i) = mean(f_non(i,:));
% end
% 
% save('nonface_feat_mean3.mat','nonface_feat_mean','-mat','-v7.3');
% 

% now compute mean of positive and negative samples and the parity

% load('face_feat_mean.mat');
% 
% load('nonface_feat_mean1.mat');
% x1 = nonface_feat_mean;
% 
% load('nonface_feat_mean2.mat');
% x2 = nonface_feat_mean;
% 
% load('nonface_feat_mean3.mat');
% x3 = nonface_feat_mean;
% 
% threshold = zeros(1,100000);
% threshold_sign = zeros(1,100000);
% 
% for i = 1:100000
%     mean_val = (face_feat_mean(i)+x1(i)+x2(i)+x3(i))/4;
%     if(face_feat_mean(i) > threshold)
%         threshold_sign(i) = 1;
%     else
%         threshold_sign(i) = -1;
%     end
%     threshold(i) = mean_val;
% end
% 
% save('threshold_sign.mat','threshold_sign','-mat','-v7.3');
% save('threshold.mat','threshold','-mat','-v7.3');

load('nonface_feat3.mat');
nonface_samples3 = f_non(1:5000,1:3000);
save('nonface_samples3.mat','nonface_samples3','-mat','-v7.3');