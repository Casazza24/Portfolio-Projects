%% spectral_analysis.m — FFT-based periodicity analysis of PJM East energy demand
%
% Reads cleaned hourly energy data, computes the power spectral density via
% FFT, identifies dominant cyclical periods (daily, weekly, annual), and
% extracts spectral features (sin/cos amplitude pairs for top frequency
% components) for use as engineered features in the forecasting model.
%
% Usage (from the project root):
%     matlab -batch "run('spectral_analysis.m')"
%
% Inputs:  data/pjm_clean.csv    (Datetime, PJME_MW)
% Outputs: data/spectral_features.csv   (one row per timestamp)
%          outputs/psd_plot.png          (power spectral density plot)

%% ---- Configuration ----
data_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
out_dir  = fullfile(fileparts(mfilename('fullpath')), 'outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ---- 1. Load data ----
fprintf('Loading pjm_clean.csv ...\n');
T = readtable(fullfile(data_dir, 'pjm_clean.csv'));
timestamps = datetime(T.Datetime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
demand     = T.PJME_MW;
N          = length(demand);
Fs         = 1;  % sampling rate: 1 sample per hour
fprintf('  %d hourly observations, %s to %s\n', N, ...
    datestr(timestamps(1)), datestr(timestamps(end)));

%% ---- 2. FFT ----
% Remove mean (DC component) so the spectrum focuses on oscillations
demand_centered = demand - mean(demand);

Y    = fft(demand_centered);
P2   = abs(Y / N);                    % two-sided amplitude spectrum
P1   = P2(1:floor(N/2)+1);           % one-sided
P1(2:end-1) = 2 * P1(2:end-1);       % double non-DC/Nyquist bins
PSD  = P1 .^ 2;                       % power spectral density (amplitude^2)

freq = Fs * (0:floor(N/2)) / N;       % frequency axis (cycles/hour)
period_hours = 1 ./ freq;             % convert to period in hours

%% ---- 3. Identify dominant periods ----
% Focus on periods between 6 hours and 10000 hours (just over a year)
valid = freq > 0 & period_hours >= 6 & period_hours <= 10000;
[~, sorted_idx] = sort(PSD(valid), 'descend');
valid_indices = find(valid);

% Top 10 peaks for display, top 3 for feature extraction
n_display = min(10, length(sorted_idx));
n_features = 3;
top_display = valid_indices(sorted_idx(1:n_display));
top_feat    = valid_indices(sorted_idx(1:n_features));

fprintf('\n=== Top %d Dominant Periods ===\n', n_display);
fprintf('  %-6s  %-12s  %-10s\n', 'Rank', 'Period (h)', 'Power');
for k = 1:n_display
    idx = top_display(k);
    p_h = period_hours(idx);
    % Annotate recognized cycles
    label = '';
    if abs(p_h - 24) < 2,       label = '  <-- daily';
    elseif abs(p_h - 168) < 10,  label = '  <-- weekly';
    elseif abs(p_h - 8760) < 500, label = '  <-- ~annual';
    elseif abs(p_h - 12) < 1,    label = '  <-- semi-daily';
    end
    fprintf('  %-6d  %-12.1f  %-10.2e%s\n', k, p_h, PSD(idx), label);
end

%% ---- 4. Extract spectral features ----
% For each of the top 3 frequency components, compute sin/cos pairs at
% every timestamp. These encode both amplitude and phase as continuous
% features suitable for ML models (avoiding the discontinuity of raw
% phase angles at 0/2pi).
%
% For frequency f_k with complex FFT coefficient Y(k):
%   amplitude = 2*|Y(k)|/N
%   phase     = angle(Y(k))
%   feature_sin(t) = amplitude * sin(2*pi*f_k*t + phase)
%   feature_cos(t) = amplitude * cos(2*pi*f_k*t + phase)
%
% where t is hours since the start of the series.

fprintf('\nExtracting spectral features for top %d components ...\n', n_features);
t_hours = hours(timestamps - timestamps(1));  % time axis in hours

feat_names = {};
feat_data  = zeros(N, 2 * n_features);

for k = 1:n_features
    fft_idx = top_feat(k);
    f_k     = freq(fft_idx);
    amp     = 2 * abs(Y(fft_idx)) / N;
    phi     = angle(Y(fft_idx));
    p_h     = period_hours(fft_idx);

    % Name by approximate recognized period
    if abs(p_h - 24) < 2,       name = 'daily';
    elseif abs(p_h - 168) < 10, name = 'weekly';
    elseif abs(p_h - 8760) < 500, name = 'annual';
    elseif abs(p_h - 12) < 1,   name = 'semidaily';
    else,                        name = sprintf('period_%dh', round(p_h));
    end

    sin_col = sprintf('spectral_%s_sin', name);
    cos_col = sprintf('spectral_%s_cos', name);
    feat_names{end+1} = sin_col;
    feat_names{end+1} = cos_col;

    col = 2*(k-1);
    feat_data(:, col+1) = amp * sin(2*pi*f_k*t_hours + phi);
    feat_data(:, col+2) = amp * cos(2*pi*f_k*t_hours + phi);

    fprintf('  Component %d: period=%.1fh (%s), amplitude=%.1f MW\n', ...
        k, p_h, name, amp);
end

%% ---- 5. Save spectral features CSV ----
out_table = table(timestamps, 'VariableNames', {'Datetime'});
for k = 1:length(feat_names)
    out_table.(feat_names{k}) = feat_data(:, k);
end

out_path = fullfile(data_dir, 'spectral_features.csv');
writetable(out_table, out_path);
fprintf('\nSaved spectral features: %s (%d rows, %d feature columns)\n', ...
    out_path, N, length(feat_names));

%% ---- 6. PSD plot ----
fig = figure('Visible', 'off', 'Position', [100 100 1000 500]);

% Plot PSD vs period (more intuitive than vs frequency)
loglog(period_hours(valid), PSD(valid), 'Color', [0.3 0.45 0.69], 'LineWidth', 0.8);
hold on;

% Mark the top 3 peaks
for k = 1:n_features
    idx = top_feat(k);
    loglog(period_hours(idx), PSD(idx), 'ro', 'MarkerSize', 10, ...
        'MarkerFaceColor', [0.89 0.33 0.34], 'LineWidth', 1.5);
    % Label offset slightly above
    text(period_hours(idx), PSD(idx)*1.8, ...
        sprintf('%.0fh', period_hours(idx)), ...
        'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold');
end

% Reference lines for known cycles
for ref = [24, 168, 8760]
    xline(ref, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8);
end
text(24, min(PSD(valid))*3, '24h (daily)', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
text(168, min(PSD(valid))*3, '168h (weekly)', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
text(8760, min(PSD(valid))*3, '8760h (annual)', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);

xlabel('Period (hours)', 'FontSize', 12);
ylabel('Power Spectral Density', 'FontSize', 12);
title('PJM East Energy Demand — Dominant Cyclical Periods', 'FontSize', 14);
grid on;
hold off;

plot_path = fullfile(out_dir, 'psd_plot.png');
exportgraphics(fig, plot_path, 'Resolution', 150);
fprintf('Saved PSD plot: %s\n', plot_path);

fprintf('\nDone. Next: run analysis.py to build the forecasting model.\n');
