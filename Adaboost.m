function[] = Adaboost()

load('face_samples.mat');
load('nonface_samples1.mat');
load('nonface_samples2.mat');
load('nonface_samples3.mat');
load('non_face_samples4.mat');

load('threshold.mat');
load('threshold_sign.mat');

tr_data = [face_samples' ; nonface_samples1'; nonface_samples2'; nonface_samples3'; non_face_samples4'];
tr_threshold = threshold;
tr_class = [ones(1,5000), (-1)*ones(1,9000+5437)];
tr_threshold_sign = threshold_sign;

% nXd
%tr_data = [1 1 1; 3 3 3; 2 2 2; 4 4 4];
%tr_threshold = [2.5, 3.5, 2];
%tr_class = [1, 1, -1,-1];

% to decide +ve sample is less or greater than threshold(mean) value
%tr_threshold_sign = [1, -1, 1];

%output class: face (+1) vs non-face (-1)
class = [1, -1];
used_classifiers = zeros(1,size(tr_data,2));

D = ones(1,size(tr_data,1))*(1.0/size(tr_data,1));
epsilon = zeros(1,size(tr_data,2));
epsilon_copy = zeros(1,size(tr_data,2));
classifier_weights = zeros(1,size(tr_data,2));

for t = 1:100
    t
    if((t == 2) || (t ==10) || (t==50) || (t==100) || (t==200)) 
        save(['classifier_weights_', num2str(t), '.mat'],'classifier_weights','-mat','-v7.3');
        save(['epsilon_copy', num2str(t), '.mat'],'epsilon_copy','-mat','-v7.3');
    end
    for j = 1:size(tr_data,2)
        if(used_classifiers(j))
            continue;
        end
        error = 0;
        for k = 1:size(tr_data,1)
            d = D(k);
            tr_feat_val = tr_data(k,j);
            
            if(tr_threshold_sign(j) == 1)
                if ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == 1))
                    error = error + 0*d;
                elseif ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == -1))
                    error = error + 0*d;
                else
                    error = error + 1*d;
                end
            end
            if(tr_threshold_sign(j) == -1)
                if ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == 1))
                    error = error + 0*d;
                elseif ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == -1))
                    error = error + 0*d;
                else
                    error = error + 1*d;
                end
            end
            
        end
        epsilon(j) =  error;
        epsilon_copy(j) =  error;
    end
    
    [min_val, min_index] = min(epsilon);
    
    if(min_val == Inf)
        break;
    end
    
    alpha_t = (1.0/2)*(log((1-min_val)/min_val));
    
    classifier_weights(min_index) = alpha_t;
    
    for k = 1:size(tr_data,1)
        tr_feat_val = tr_data(k,min_index);
        
        h_class = 0;
        if(tr_threshold_sign(min_index) == 1)
            if (tr_feat_val > tr_threshold(min_index))
                h_class = 1;
                
            elseif (tr_feat_val <= tr_threshold(min_index))
                h_class = -1;
            end
        end
        if(tr_threshold_sign(min_index) == -1)
            if (tr_feat_val < tr_threshold(min_index))
                h_class = 1;
                
            elseif (tr_feat_val >= tr_threshold(min_index))
                h_class = -1;
            end
        end
        D(k) = D(k)*exp(-1*alpha_t*tr_class(k)*h_class);
    end
    
    D = D/sum(D);
    epsilon(min_index) = Inf;
    used_classifiers(min_index) = 1;
end

%save('classifier_weights.mat','classifier_weights','-mat','-v7.3');
%save('top_errors_T_0.mat','errors_T_0','-mat','-v7.3');
%save('top_errors_T_10.mat','errors_T_10','-mat','-v7.3');
%save('top_errors_T_50.mat','errors_T_50','-mat','-v7.3');
%save('top_errors_T_100.mat','errors_T_100','-mat','-v7.3');
