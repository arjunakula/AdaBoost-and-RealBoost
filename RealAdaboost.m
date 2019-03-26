function[] = RealAdaboost()

load('face_samples.mat');
load('nonface_samples1.mat');
load('nonface_samples2.mat');
load('nonface_samples3.mat');

load('threshold.mat');
load('threshold_sign.mat');

bins = linspace(min(threshold),max(threshold),19);

tr_data = [face_samples' ; nonface_samples1'; nonface_samples2'; nonface_samples3'];
tr_threshold = threshold;
tr_class = [ones(1,5000), (-1)*ones(1,9000)];
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

p_t =  zeros(20,size(tr_data,2));
q_t = zeros(20,size(tr_data,2));

for t = 1:100
    t
    if((t == 2) || (t ==10) || (t==50) || (t==100) || (t==200)) 
        save(['pt_', num2str(t), '.mat'],'p_t','-mat','-v7.3');
        save(['qt_', num2str(t), '.mat'],'q_t','-mat','-v7.3');
        save(['bins_', num2str(t), '.mat'],'bins','-mat','-v7.3');
        %save(['classifier_weights_', num2str(t), '.mat'],'classifier_weights','-mat','-v7.3');
        %save(['epsilon_copy', num2str(t), '.mat'],'epsilon_copy','-mat','-v7.3');
    end
    for j = 1:size(tr_data,2)
        if(used_classifiers(j))
            continue;
        end
        %error = 0;
        for k = 1:size(tr_data,1)
            d = D(k);
            tr_feat_val = tr_data(k,j);
            
            bin_id = 1;
            for bi = 1:19
                if(tr_feat_val >= bins(bi))
                    bin_id = bi+1;
                end
            end
            
            if(tr_class(k) == 1)
                p_t(bin_id, j) = p_t(bin_id,j) + d;
            else
                q_t(bin_id, j) = q_t(bin_id, j)+d;
            end
            
%             if(tr_threshold_sign(j) == 1)
%                 if ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == 1))
%                     error = error + 0*d;
%                 elseif ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == -1))
%                     error = error + 0*d;
%                 else
%                     error = error + 1*d;
%                 end
%             end
%             if(tr_threshold_sign(j) == -1)
%                 if ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == 1))
%                     error = error + 0*d;
%                 elseif ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == -1))
%                     error = error + 0*d;
%                 else
%                     error = error + 1*d;
%                 end
%             end
%             
        end
%         epsilon(j) =  error;
%         epsilon_copy(j) =  error;
    end
    
    Z_val = zeros(1,size(tr_data,2));
    for j = 1:size(tr_data,2)
        if(used_classifiers(j))
            continue;
        end
        bsum = 0;
        for bi = 1:20
            bsum = bsum + sqrt(p_t(bi,j)*q_t(bi,j));
        end
        Z_val(j) = 2*bsum;
    end
    
    [min_val, min_index] = min(Z_val);
    
    if(min_val == Inf)
        break;
    end
    
   %alpha_t = (1.0/2)*(log((1-min_val)/min_val));
    
    %classifier_weights(min_index) = alpha_t;
    
    for k = 1:size(tr_data,1)
        tr_feat_val = tr_data(k,min_index);
        
         bin_id = 1;
            for bi = 1:19
                if(tr_feat_val >= bins(bi))
                    bin_id = bi+1;
                end
            end
            
            h_tb = 1/2*log(p_t(bin_id,min_index)/q_t(bin_id,min_index));
        
%         h_class = 0;
%         if(tr_threshold_sign(min_index) == 1)
%             if (tr_feat_val > tr_threshold(min_index))
%                 h_class = 1;
%                 
%             elseif (tr_feat_val <= tr_threshold(min_index))
%                 h_class = -1;
%             end
%         end
%         if(tr_threshold_sign(min_index) == -1)
%             if (tr_feat_val < tr_threshold(min_index))
%                 h_class = 1;
%                 
%             elseif (tr_feat_val >= tr_threshold(min_index))
%                 h_class = -1;
%             end
%         end
        D(k) = D(k)*exp(-1*tr_class(k)*h_tb);
    end
    
    D = D/sum(D);
    %epsilon(min_index) = Inf;
    used_classifiers(min_index) = 1;
end

save('pt.mat','p_t','-mat','-v7.3');
save('qt.mat','q_t','-mat','-v7.3');
save('bins.mat','bins','-mat','-v7.3');

%save('classifier_weights.mat','classifier_weights','-mat','-v7.3');
%save('top_errors_T_0.mat','errors_T_0','-mat','-v7.3');
%save('top_errors_T_10.mat','errors_T_10','-mat','-v7.3');
%save('top_errors_T_50.mat','errors_T_50','-mat','-v7.3');
%save('top_errors_T_100.mat','errors_T_100','-mat','-v7.3');
