 load('face_samples.mat');
    load('nonface_samples1.mat');
    load('nonface_samples2.mat');
    load('nonface_samples3.mat');

    load('threshold.mat');
    load('threshold_sign.mat');

    load('classifier_weights_100.mat');
    load('pt_100.mat');
    load('qt_100.mat');
    load('bins_2.mat')

    tr_data = [face_samples' ; nonface_samples1'; nonface_samples2'; nonface_samples3'];
    tr_threshold = threshold;
    tr_class = [ones(1,5000), (-1)*ones(1,9000)];
    tr_threshold_sign = threshold_sign;

    F_positive = [];

    F_negative = [];

    index = 1;

    for k = 1:5000
        sum = 0;
        for j = 1:size(tr_data,2)
            tr_feat_val = tr_data(k,j);
            
             bin_id = 1;
            for bi = 1:19
                if(tr_feat_val >= bins(bi))
                    bin_id = bi+1;
                end
            end
            h_tb = 1/2*log(p_t(bin_id,j)/q_t(bin_id,j));
            if(h_tb == Inf || h_tb== -Inf)
                h_tb = 0;
            end
%             decision = 0;
%             if(tr_threshold_sign(j) == 1)
%                     if ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == 1))
%                         decision = 1;
%                     elseif ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == -1))
%                         decision = 1;
%                     else
%                         decision = -1;
%                     end
%                 end
%                 if(tr_threshold_sign(j) == -1)
%                     if ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == 1))
%                         decision = 1;
%                     elseif ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == -1))
%                         decision = 1;
%                     else
%                         decision = -1;
%                     end
%                 end

            sum = sum+h_tb;
        end

            F_positive(index) = sum;
            index = index+1;

    end

    index = 1;

    for k = 5001:14000
        sum = 0;
        for j = 1:size(tr_data,2)
            tr_feat_val = tr_data(k,j);
            
            bin_id = 1;
            for bi = 1:19
                if(tr_feat_val >= bins(bi))
                    bin_id = bi+1;
                end
            end
            h_tb = 1/2*log(p_t(bin_id,j)/q_t(bin_id,j));
            if(h_tb == Inf || h_tb== -Inf)
                h_tb = 0;
            end
            
%             decision = 0;
%             if(tr_threshold_sign(j) == 1)
%                     if ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == 1))
%                         decision = 1;
%                     elseif ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == -1))
%                         decision = 1;
%                     else
%                         decision = -1;
%                     end
%                 end
%                 if(tr_threshold_sign(j) == -1)
%                     if ((tr_feat_val < tr_threshold(j)) && (tr_class(k) == 1))
%                         decision = 1;
%                     elseif ((tr_feat_val > tr_threshold(j)) && (tr_class(k) == -1))
%                         decision = 1;
%                     else
%                         decision = -1;
%                     end
%                 end

            sum = sum+h_tb;
        end

            F_negative(index) = sum;

            index = index+1;

    end

    F_positive = [F_positive, F_positive];
    F_negative = [F_negative, F_negative, F_negative, F_negative];

 h1 = histfit(F_positive);
 h1(1).FaceColor = [.8 .8 1];
 h1(2).Color = [0.1 0.2 .8];
 
 hold on;
 
 h2 = histfit(F_negative);
 h2(1).FaceColor = [1 .7 .7];
 h2(2).Color = [0.0 0.2 .1];
 
 title('Negtative population on the left, positive population on right');
 xlabel('F(x)')
 
%  % ROC curves
%  labels = [ones(1,size(F_positive,2)), ones(1,size(F_negative,2))*-1];
%  scores = [F_positive, F_negative];
%  posclass = 1;
%  [X,Y] = perfcurve(labels, scores,posclass);
%  plot(X,Y);
%  title('ROC curve');