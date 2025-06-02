% Load raw data (assumes variable `data` exists)
load('data_1s.mat');  % size assumed to be [time x channels]

% Compute temporal difference (highlights signal changes)
delta = diff(data);  % size = [time-1 x channels]

% Convert to double for precision
X_original = double(delta);

% Z-score normalization
mean_X = mean(X_original);
std_X = std(X_original);
std_X(std_X == 0) = 1e-6;  % to avoid division by zero

X = (X_original - mean_X) ./ std_X;

% Save for training
save('preprocessed_full_data.mat', ...
     'X', 'X_original', 'mean_X', 'std_X');
