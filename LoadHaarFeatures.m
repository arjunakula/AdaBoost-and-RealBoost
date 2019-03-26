function [] = LoadHaarFeatures()
clear all;

myFolder = 'newface16/';
filePattern = fullfile(myFolder, '*.bmp');
datFiles = dir(filePattern);

fprintf('%d training images\n', length(datFiles));
fprintf('images processed =');

 %f_non = zeros(100000,length(datFiles));
f_non = zeros(100000,15356);

for nn = 1:length(datFiles)
%for nn = 30001:45356
        
    if mod(nn,5000) == 0
        fprintf(' %d\n', nn);
    end
    
    
    baseFileName = datFiles(nn).name;
    fullFileName = fullfile(myFolder, baseFileName);
    
    img = imread(fullFileName);
    
    img = rgb2gray(img);
    
    integ_image = integralImage(img);
    
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
                    if(feat_counter > 100000)
                        break;
                    end
                     if(feat_counter == 1785)
                         x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6
                         return;
                         break;
                    end
                    %feat_counter,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6
                    %-integ_image(y1,x1)+integ_image(y2,x2)+2*integ_image(y3,x3)-2*integ_image(y4,x4)-integ_image(y5,x5)+integ_image(y6,x6)
                    f_non(feat_counter,nn) = -integ_image(y1,x1)+integ_image(y2,x2)+2*integ_image(y3,x3)-2*integ_image(y4,x4)-integ_image(y5,x5)+integ_image(y6,x6);
                end
                
                if(feat_counter > 100000)
                    break;
                end
                
            end
            
            if(feat_counter > 100000)
                break;
            end
            
        end
        
        if(feat_counter > 100000)
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
                    if(feat_counter > 100000)
                        break;
                    end
                    f_non(feat_counter,nn) = -integ_image(y1,x1)+integ_image(y2,x2)+2*integ_image(y3,x3)-2*integ_image(y4,x4)-integ_image(y5,x5)+integ_image(y6,x6);
                end
                if(feat_counter > 100000)
                    break;
                end
            end
            if(feat_counter > 100000)
                break;
            end
        end
        if(feat_counter > 100000)
            break;
        end
    end
    
end


%save('nonface_feat3.mat','f_non','-mat','-v7.3');