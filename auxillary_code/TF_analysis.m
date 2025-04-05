
%% Perform time-frequency analysis 
%-------------------------------------------------------
%% Load EEG data
load('C:\Users\abete\Dropbox\CuttingGardens\data\s04_JADE_trimmed_ds.mat')
EEG.data = double(EEG.data);

%% Cleaning the data
ICs2remove = [1 2]; % blink and non-brain activity ICs
EEG = pop_subcomp(EEG,ICs2remove);

%% Wavelet TF analysis
% Wavelet parameters
times2save = -300:25:3200;
basetime   = [ -500 -200 ];

min_frex  =  2;
max_frex  = 30; % was 40
num_frex  = 40; % was 40
nCycRange = [4 6]; % was [4 12]

% setup convolution parameters
times2saveidx = dsearchn(EEG.times',times2save');
baseidx = dsearchn(EEG.times',basetime');

frex = logspace(log10(min_frex),log10(max_frex),num_frex);

% wavelet parameters
s = logspace(log10(nCycRange(1)),log10(nCycRange(2)),num_frex)./(2*pi.*frex);
t = -2:1/EEG.srate:2;
halfwave = floor((length(t)-1)/2);

nData = EEG.trials*EEG.pnts;
nWave = length(t);
nConv = nData+nWave-1;

% wavelets
cmwX = zeros(num_frex,nConv);
for fi=1:num_frex
    cmw = fft(  exp(1i*2*pi*frex(fi).*t) .* exp( (-t.^2)/(2*s(fi)^2) )  ,nConv);
    cmwX(fi,:) = cmw ./ max(cmw);
end

% initialize trial-average output matrices
tf_all = zeros(EEG.nbchan,num_frex,length(times2save));
tf_cond = zeros(EEG.nbchan,length(condlist),num_frex,length(times2save));

% fft of data at FCz (47) Try Cz(48)
for chani = 1:EEG.nbchan
    disp(chani)
    eegX = fft( reshape(EEG.data(chani,:,:),1,nData) ,nConv);
    
    % loop over frequencies
    for fi=1:num_frex
        as = ifft( eegX.*cmwX(fi,:) );
        as = as(halfwave+1:end-halfwave);
        as = reshape(as,EEG.pnts,EEG.trials);
        
        % condition-average power baselined power
        basepow = mean(mean( abs(as(baseidx(1):baseidx(2),:)).^2,2),1);
        tf_all(chani,fi,:) = 10*log10( mean(abs(as(times2saveidx,:)).^2,2) ./ basepow );
        
        % condition-specific, trial-average power
        for condi=1:length(condlist)
            tf_cond(chani,condi,fi,:) = 10*log10( mean( abs(as(times2saveidx,condlist(condi)==condmarks)).^2 ,2) / basepow );
        end
        
    end % end frequencies loop
end

chanlocs = EEG.chanlocs;
outfilename = [homedir 's04_TF_2_30Hz_40freqs_ds.mat'];
save(outfilename,'tf_all','tf_cond','frex','times2save','condmarks','condlist','nCycRange','chanlocs');

%% END