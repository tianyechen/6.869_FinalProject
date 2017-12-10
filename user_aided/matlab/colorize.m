clear all

original = double(imread('example.bmp'))/255;
marked   = double(imread('example_marked.bmp'))/255;
out_name='example_res_output.bmp';

% sum(X,dim) sums along the dimesions dim, i.e. sum(..,3) for image
% sums in the third dimension matrix columns and outputs vector with
% elements of the sums of each column
% Then takes all the colored columns as colorIm
threshold = 0.01;
colorIm = sum(abs(original - marked), 3) > threshold;
colorIm = double(colorIm);

% As the algorithm works in YUV color mode where Y is the grayscale value
% and U, V define color, rgb2ntsc is used (converts to YUV basically)
YIQ_gray = my_rgb2ntsc(original);
YIQ_color = my_rgb2ntsc(marked);

% Make a new image with Y as the grayscale first dimension
% and other dimensions from the color image

YUV(:,:,1) = YIQ_gray(:,:,1);
YUV(:,:,2) = YIQ_color(:,:,2);
YUV(:,:,3) = YIQ_color(:,:,3);

% Takes log from the min of the first two dimension sizes of the new image
% divides it by the log(2), substracts 2 and takes floor
% then names it "max dimension" :D
% max_d = floor(log(min(size(YUV,1),size(YUV,2)))/log(2)-2);
% % does some magic
% iu = floor(size(YUV,1)/(2^(max_d-1)))*(2^(max_d-1));
% ju = floor(size(YUV,2)/(2^(max_d-1)))*(2^(max_d-1));
% id = 1;
% jd = 1;
% % specifies colorIm pixels that are painted I guess
% colorIm = colorIm(id:iu,jd:ju,:);
% YUV = YUV(id:iu,jd:ju,:);

% solver 1 ie matlab itself by default
colorizedIm = abs(getColorExact(colorIm,YUV));

figure, image(colorizedIm)

% write the image to file
%imwrite(nI,out_name)
