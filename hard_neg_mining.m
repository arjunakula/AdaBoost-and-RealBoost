load('face_samples.mat');
load('nonface_samples1.mat');
load('nonface_samples2.mat');
load('nonface_samples3.mat');

load('threshold.mat');
load('threshold_sign.mat');

load('classifier_weights_100.mat');
[sortedX,sortingIndices] = sort(classifier_weights,'descend');

myFolder = 'Test_and_background_Images/';
filePattern = fullfile(myFolder, '*.jpg');
datFiles = dir(filePattern);

non_face_samples4 = [];

for nn = 1:length(datFiles)
    
    baseFileName = datFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    
    img = imread(fullFileName);
    
    img = rgb2gray(img);
    img = imresize(img,1);
    
    size(img);
    
    integ_image_full = integralImage(img);
    
    % divide the test negative image to 16X16 pixels and convolute.
    
    img_id = 1;
    for conv_row = 1:16:(size(img,1)-16)
        nn,conv_row,size(non_face_samples4)
        for conv_col = 1:16:(size(img,2)-16)
            
            integ_image = integ_image_full(conv_row:conv_row+16,conv_col:conv_col+16);
    f_non = zeros(5000,1);
    feat_counter = 0;
    for h = 1:16
        for w = 1:8
            for i = 1:17-h
                for j = 1:17-2*w
                    x1 = j;
                    x2 = j;
                    x3 = j+w;
                    x4 = j+w;
                    x5 = j+2*w;
                    x6 = j+2*w;
                    y1 = i;
                    y3 = i;
                    y5 = i;
                    y2 = i+h;
                    y4 = i+h;
                    y6 = i+h;
                    
                    feat_counter = feat_counter+1;
                    if(feat_counter > 5000)
                        break;
                    end
%                      if(feat_counter == 1785)
%                          x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6
%                          return;
%                          break;
%                     end
                    %feat_counter,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6
                    %-integ_image(y1,x1)+integ_image(y2,x2)+2*integ_image(y3,x3)-2*integ_image(y4,x4)-integ_image(y5,x5)+integ_image(y6,x6)
                    f_non(feat_counter,1) = -integ_image(y1,x1)+integ_image(y2,x2)+2*integ_image(y3,x3)-2*integ_image(y4,x4)-integ_image(y5,x5)+integ_image(y6,x6);
                end
                
                if(feat_counter > 5000)
                    break;
                end
                
            end
            
            if(feat_counter > 5000)
                break;
            end
            
        end
        
        if(feat_counter > 5000)
            break;
        end
        
    end
    
    for h = 1:8
        for w = 1:16
            for i = 1:17-2*h
                for j = 1:17-w
                    x1 = j;
                    x3 = j;
                    x5 = j;
                    x2 = j+w;
                    x4 = j+w;
                    x6 = j+w;
                    y1 = i;
                    y2 = i;
                    y3 = i+h;
                    y4 = i+h;
                    y5 = i+2*h;
                    y6 = i+2*h;
                    
                    feat_counter = feat_counter+1;
                    if(feat_counter > 5000)
                        break;
                    end
                    f_non(feat_counter,1) = -integ_image(y1,x1)+integ_image(y2,x2)+2*integ_image(y3,x3)-2*integ_image(y4,x4)-integ_image(y5,x5)+integ_image(y6,x6);
                end
                if(feat_counter > 5000)
                    break;
                end
            end
            if(feat_counter > 5000)
                break;
            end
        end
        if(feat_counter > 5000)
            break;
        end
    end
    
    F_x_sum = 0;
    for ssi = 1:100
        cwt = sortingIndices(ssi);
        if(cwt > 5000)
            cwt = 808;
        end
         h_class = 0;
         tr_feat_val = f_non(cwt,1);
        if(threshold_sign(cwt) == 1)
            if (tr_feat_val > threshold(cwt))
                h_class = 1;
                
            elseif (tr_feat_val <= threshold(cwt))
                h_class = -1;
            end
        end
        if(threshold_sign(cwt) == -1)
            if (tr_feat_val < threshold(cwt))
                h_class = 1;
                
            elseif (tr_feat_val >= threshold(cwt))
                h_class = -1;
            end
        end
        
        tmp = classifier_weights(cwt)*h_class;
        F_x_sum = F_x_sum + tmp;
    end
    
    if(F_x_sum > 0)
    non_face_samples4 = [non_face_samples4, f_non];
    end
        end
    end
    
    if(size(non_face_samples4,2) > 5000)
        break
    end
    
end
 save('non_face_samples4.mat','non_face_samples4','-mat','-v7.3');