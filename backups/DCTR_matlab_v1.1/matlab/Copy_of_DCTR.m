function [F DCT_res] = DCTR(I_STRUCT, QF)

%% Set parameters
% number of histogram bins
T = 4;

% compute quantization step based on quality factor
if QF<50,
    q = min(8 * (50 / QF), 100);
else
    q = max(8 * (2 - (QF/50)), 0.2);
end

%% Prepare for computing DCT bases
k=0:4;
l=0:4;
[k,l]=meshgrid(k,l);

A=cos(((2.*k+1).*l*pi)/10)./sqrt(5);
A(1,:)=A(1,:)./sqrt(2);
A = A.*sqrt(2);
A=A';

%% Compute DCTR locations to be merged
mergedCoordinates = cell(25, 1);
for i=1:5
    for j=1:5
        coordinates = [i,j; i,10-j; 10-i,j; 10-i,10-j];
        coordinates = coordinates(all(coordinates<9, 2), :);
        mergedCoordinates{(i-1)*5 + j} = unique(coordinates, 'rows');
    end
end

% Decompress to spatial domain
fun = @(x)x.data .*I_STRUCT.quant_tables{1};
I_spatial = blockproc(I_STRUCT.coef_arrays{1},[8 8],fun);
fun=@(x)idct2(x.data);
I_spatial = blockproc(I_spatial,[8 8],fun)+128;

%% Compute features
modeFeaDim = numel(mergedCoordinates)*(T+1);
F = zeros(1, 64*modeFeaDim, 'single');
DCT_res = zeros(5, 5, 25);
for mode_r = 1:5
    for mode_c = 1:5
        modeIndex = (mode_r-1)*5 + mode_c;
        % Get DCT base for current mode
        DCTbase = A(:,mode_r)*A(:,mode_c)';
        
        DCT_res(:, :, modeIndex) = DCTbase;
    end
end

end

