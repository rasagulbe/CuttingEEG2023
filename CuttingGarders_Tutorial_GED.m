%% CuttingGardens Tutorial 2023 Oct, Frankfurt

% Rasa Gulbinaite. Email: rasa.gulbinaite@gmail.com

% The code is based on several papers and code adapted from the book:

%     1. Gulbinaite et al. (2017). JoN (https://pubmed.ncbi.nlm.nih.gov/28931569/)
%     2. Cohen & Gulbinaite (2016). NeuroImage (https://pubmed.ncbi.nlm.nih.gov/27916666/)
%     3. Mike X Cohen, MIT Press (2014) "Analyzing Neural Time Series Data"

%--------------------------------------------------------------------------
%%  DATASET: Cognitive control task + flicker (Gulbinaite et al., JoN 2017)
%
%--------------------------------------------------------------------------
% DESCRIPTION of the TASK: see the PPT slides
%
% ORGANIZATION OF THE DATA:
% EEG structure - data in EEGLab format (channels x time x trials)
% bd - behavioral data
%--------------------------------------------------------------------------

%% Start with the white page
set(0,'DefaultFigureWindowStyle','docked')
% set(0,'DefaultFigureWindowStyle','normal')

clear all
close all
%% Add EEG lab to the path and define directory with the data/scripts/EEGlab toolbox

% Define the general path were everything resides
homedir = 'C:\Users\abete\Dropbox\CuttingGardens\';
cd(homedir);

addpath(genpath([homedir '\eeglab2021.0\']))        % add EEGlab toolbox in the path        
addpath([homedir '\BrewerMap-master'])              % add nice colormaps to the path
cmap = flipud(colormap(brewermap(256, 'RdYlBu')));  % load the colormap
close

%% 0A. Explore the data (TF results)

load([homedir '\data\s04_TF_2_30Hz_40freqs_ds.mat'])   % load the data

%% Plot topographies of theta (3-6 Hz) and alpha (8-12 Hz) frequency bands

freqs2plot = [3 6; 8 12];
plotwin = [0 2500];
tstep = 250;
wins(1,:) = plotwin(1):tstep:plotwin(2);
wins(2,:) = plotwin(1)+tstep:tstep:plotwin(2)+tstep;

freqs2plot_idx = zeros(size(freqs2plot)); wins_idx = zeros(size(wins));
for i = 1:size(wins,1)
    freqs2plot_idx(i,:) = dsearchn(frex',freqs2plot(i,:)');
    wins_idx(i,:) = dsearchn(times2save',wins(i,:)');
end
numcols = size(wins,2);
numrows = size(wins,1);

figure, set(gcf,'Name','General task overview')

for rowi=1:numrows

    for coli=1:numcols

        subplot(numrows,numcols,(rowi-1)*numcols+coli)
        data2plot = squeeze(mean(mean(tf_all(:,freqs2plot_idx(rowi,1):freqs2plot_idx(rowi,2),wins_idx(1,coli):wins_idx(2,coli)),2),3));
        topoplot(data2plot,chanlocs,'plotrad',.7,'electrodes','off','numcontour',0,'whitebk','on');%,'emarker',{'.','k',15,3});

        % titles in the plot
        if rowi == 1
            title([ num2str(wins(1,coli)) '-' num2str(wins(2,coli)) ' ms' ])
        end
        if coli==1 && rowi==1
            h = text(-2.5, 0, ['Power ' num2str(freqs2plot(rowi,1)) '-' num2str(freqs2plot(rowi,2)) ' Hz' ]);
        elseif coli==1 && rowi==2
            h = text(-2.5, 0, ['Power ' num2str(freqs2plot(rowi,1)) '-' num2str(freqs2plot(rowi,2)) ' Hz' ]);
        end
    end
end
set(gcf,'color','w')
colormap(cmap)
clear data2plot
%% Plot TF decomposition results:
%     a) Condition-average/condition-specific theta power
%     b) Conflict effect

conditionLabels = { '10/7 I';'10/7 C';'7/10 I';'7/10 C';'15/10 I';'15/10 C';'10/15 I';'10/15 C' };

% Which channel to plot?
chan2plot = 'FCz'; % Try Cz channel
chan2plot_idx = find(strcmpi(chan2plot,{chanlocs.labels}));

% Find Incongruent (conflict) & Congruent (no-conflict) trial condition indices
inc_idx = find(~cellfun(@isempty,strfind(conditionLabels,' I')));
con_idx = find(~cellfun(@isempty,strfind(conditionLabels,' C')));
dat2plot_inc = squeeze(mean(tf_cond(chan2plot_idx,inc_idx,:,:),2));
dat2plot_con = squeeze(mean(tf_cond(chan2plot_idx,con_idx,:,:),2));

% Something to make plots nicer
stimtime = [0 2000];
stimtimeidx = dsearchn(times2save',stimtime')';

figure, set(gcf,'Name','TF results')
subplot(2,2,1)
contourf(times2save,frex,squeeze(tf_all(chan2plot_idx,:,:)),40,'linecolor','none')
hold on
plot([times2save(stimtimeidx(1)) times2save(stimtimeidx(1))], [frex(1) frex(end)],':k','Linewidth',1.5)
plot([times2save(stimtimeidx(2)) times2save(stimtimeidx(2))], [frex(1) frex(end)],':k','Linewidth',1.5)
ylabel('Frequencies (Hz)')
xlabel('Time (ms)')
title(['All trials, ' chan2plot ' channel'])
colormap(cmap)

subplot(2,2,2)
contourf(times2save,frex,dat2plot_inc-dat2plot_con,40,'linecolor','none')
hold on
plot([times2save(stimtimeidx(1)) times2save(stimtimeidx(1))], [frex(1) frex(end)],':k','Linewidth',1.5)
plot([times2save(stimtimeidx(2)) times2save(stimtimeidx(2))], [frex(1) frex(end)],':k','Linewidth',1.5)
ylabel('Frequencies (Hz)')
xlabel('Time (ms)')
title(['Conflict effect, ' chan2plot ' channel'])
colormap(cmap)

subplot(2,2,3)
contourf(times2save,frex,dat2plot_inc,40,'linecolor','none')
hold on
plot([times2save(stimtimeidx(1)) times2save(stimtimeidx(1))], [frex(1) frex(end)],':k','Linewidth',1.5)
plot([times2save(stimtimeidx(2)) times2save(stimtimeidx(2))], [frex(1) frex(end)],':k','Linewidth',1.5)
ylabel('Frequencies (Hz)')
xlabel('Time (ms)')
title(['Incongruent trials, ' chan2plot ' channel'])
colormap(cmap)

subplot(2,2,4)
contourf(times2save,frex,dat2plot_con,40,'linecolor','none')
hold on
plot([times2save(stimtimeidx(1)) times2save(stimtimeidx(1))], [frex(1) frex(end)],':k','Linewidth',1.5)
plot([times2save(stimtimeidx(2)) times2save(stimtimeidx(2))], [frex(1) frex(end)],':k','Linewidth',1.5)
ylabel('Frequencies (Hz)')
xlabel('Time (ms)')
title(['Congruent trials, ' chan2plot ' channel'])
colormap(cmap)

set(gcf,'color','w')

%% 0B. Explore the data: Time-frequency viewer (Mike X Cohen, MIT Press (2014) "Analyzing Neural Time Series Data")
set(0,'DefaultFigureWindowStyle','normal')
tfviewerx(times2save,frex,tf_all,chanlocs,'Cond-av, stim-locked');

%% 1. Select time and frequency window for spatial filter construction
set(0,'DefaultFigureWindowStyle','docked')

chan2plot = {'Cz'};   % Try Fcz channel
chan2plot_idx = find(strcmpi(chan2plot,{chanlocs.labels}));

% Find Theta peak (3-8 Hz)
thetatime     = [2100 3000];
thetatime_idx = dsearchn(times2save',thetatime');
thetafreq     = dsearchn(frex',[3 8]');
dat2use = squeeze(mean(tf_all(chan2plot_idx,thetafreq(1):thetafreq(2),thetatime_idx(1):thetatime_idx(2)),1));
[maxfreq,maxtime] = ind2sub(size(dat2use),find(dat2use==max(reshape(dat2use,1,[]))));
peakfreq = frex(maxfreq + thetafreq(1) - 1);
peaktime = times2save(maxtime + thetatime_idx(1) - 1);

% Plot2check
stimtime = [0 2000];
stimtimeidx = dsearchn(times2save',stimtime')';
clim = [-3 3];

figure, set(gcf,'Name','Channel TF results')
subplot(121)
topoplot(squeeze(mean(mean(tf_all(:,thetafreq(1):thetafreq(2),thetatime_idx(1):thetatime_idx(2)),2),3)),chanlocs,'plotrad',.7,'electrodes','ptslabels','numcontour',0,'whitebk','on');%,'emarker',{'.','k',15,3});
axis square
title(['Time: ' num2str(thetatime(1)) '-' num2str(thetatime(2)) ' ms'])

subplot(122)
contourf(times2save,frex,squeeze(tf_all(chan2plot_idx,:,:)),100,'linecolor','none')
hold on

plot(peaktime,peakfreq,'w*')
plot([times2save(stimtimeidx(1)) times2save(stimtimeidx(1))], [frex(1) frex(end)],':k','Linewidth',1.5)
plot([times2save(stimtimeidx(2)) times2save(stimtimeidx(2))], [frex(1) frex(end)],':k','Linewidth',1.5)
rectangle('Position',[2100 3  900 5])
set(gca,'clim',clim)
xlabel('Time (ms)'); ylabel('Frequencies (Hz)');
colorbar
colormap(cmap)
axis square
title(['TF result from ' chan2plot ' channel'])
set(gcf,'color','w')
%--------------------------------------------------------------------------
%%                 PART 2: GED for theta
%--------------------------------------------------------------------------
%       a) Discuss if it is a problem to do GED separately for congruent
%       and incongruent trials, overfitting
%       b) Trial average vs. single-trial covariance matrices

%% Load data for GED

% clear all
close all

load('C:\Users\abete\Dropbox\CuttingGardens\data\s04_JADE_trimmed_ds.mat')
EEG.data = double(EEG.data);

%% 2A. Compute covariance matrices from Concatenated trials

time4theta = [2000 2800]; % time window for theta, based on visual inspection
peakwidt   = 3;           % full-width half maximum (FWHM)
peakfreq   = 5.2871;      % hardcoded from previous analyses
tidx       = dsearchn(EEG.times',time4theta');

% Filter data around peak theta frequency
filtTmp = filterFGx(EEG.data, EEG.srate, peakfreq,peakwidt);

% Compute S covariance matrix (band-passed filtered around theta)
filtTmp  = reshape(filtTmp(:,tidx(1):tidx(2),:),EEG.nbchan,[]);  % concatenate the trials
filtTmp  = bsxfun(@minus,filtTmp,mean(filtTmp,2));               % mean-center
covTheta = (filtTmp*filtTmp') / (diff(tidx)-1);                  % compute covariance

% Compute R covariance matrix from broadband data
broadTmp = reshape(EEG.data(:,tidx(1):tidx(2),:),EEG.nbchan,[]); % concatenate the trials
broadTmp = bsxfun(@minus,broadTmp,mean(broadTmp,2));             % mean-center
covBroad = (broadTmp*broadTmp') / (diff(tidx)-1);                % compute covariance

%--------------------------------------------------------------------------
%% PLOT
%--------------------------------------------------------------------------
figure, set(gcf,'Name','Covariance matrices, concatenated')
subplot(121)
imagesc(covTheta)
axis square
title('covTheta, concatenated')
colorbar

subplot(122)
imagesc(covBroad)
axis square
title('covBroad, concatenated')
set(gcf,'color','w')
colorbar
colormap(cmap)

% Discuss rank of the covariance matrices
rank(covTheta)

%% 2B. Alternative: Compute covariance matrices from single trial data and average together

% Filter data around peak theta frequency
filtTmp = filterFGx(EEG.data, EEG.srate, peakfreq,peakwidt);

% Compute S covariance matrix
S = zeros(EEG.trials,EEG.nbchan,EEG.nbchan);
for triali = 1:EEG.trials
    trialdat = squeeze(filtTmp(:,tidx(1):tidx(2),triali));
    trialdat = bsxfun(@minus,trialdat,mean(trialdat,2));
    S(triali,:,:) = (trialdat*trialdat') / (diff(tidx)-1);
end

S = squeeze(mean(S)); % mean over trials
covTheta = S;

% Compute R covariance matrix from broadband data
R = zeros(EEG.trials,EEG.nbchan,EEG.nbchan);
for triali = 1:EEG.trials
    trialdat = squeeze(EEG.data(:,tidx(1):tidx(2),triali));
    trialdat = bsxfun(@minus,trialdat,mean(trialdat,2));
    R(triali,:,:) = (trialdat*trialdat') / (diff(tidx)-1);
end

R = squeeze(mean(R)); % mean over trials
covBroad = R;

%--------------------------------------------------------------------------
% PLOT
%--------------------------------------------------------------------------
figure, set(gcf,'Name','Covariance matrices, single-trial')
subplot(121)
imagesc(covTheta)
axis square
xlabel('Channels')
ylabel('Channels')
 set(gca,'clim',[-6 6])
% title('covTheta, single-trial')
colorbar

subplot(122)
imagesc(covBroad)
axis square
% title('covBroad, single-trial')
set(gcf,'color','w')
xlabel('Channels')
ylabel('Channels')
xlabel('Channels')
ylabel('Channels')
 set(gca,'clim',[-60 60])
colorbar
colormap(cmap)
rank(covTheta)

%% Plot channel spectra to identify bad channels - based on channel spectra in EEGlab
figure
[spectra,freqs,speccomp,contrib,specstd] = spectopo(EEG.data(:,:,:), EEG.pnts, EEG.srate,'percent',20,'limits',[2 50]);

%% 3. Remove bad channels and rerun the cells 1A or 1B (uncomment lines below)

bad_chan_idx = [17 53 61 33 1 ];
bad_channels = {EEG.chanlocs(bad_chan_idx).labels}%17 53 61 33 1

% Actual removal
EEG = pop_select(EEG,'nochannel',bad_chan_idx);

%% Run ICA without bad channels 
EEG  = pop_runica(EEG,'icatype','jader','dataset',1,'options',{20});

%% Remove bad ICs or remove bad channels?

% Compute icaact
EEG.icaact =  eeg_getica(EEG);

ICs2remove =  [1 2];

% Plot IC topgraphies
figure
for compi = 1:length(ICs2remove)
    subplot(2,length(ICs2remove),compi)
    topoplot(squeeze(EEG.icawinv(:,compi)),EEG.chanlocs, 'plotrad',.7,'electrodes','ptslabels', 'numcontour',0,'whitebk','on')
    title(['IC ' num2str(compi)])

    subplot(2,length(ICs2remove),compi+2)
%     contourf(EEG.times, 1:1:EEG.trials,squeeze(EEG.icaact(compi,:,:))',40,'linecolor','none')
    imagesc(EEG.times, [1 EEG.trials],squeeze(EEG.icaact(compi,:,:))')
    set(gca,'xlim',[-1500 3500],'ydir','normal')
    xlabel('Time (ms)')
    ylabel('Trials')
    colorbar
end
colormap(cmap)
set(gcf,'color','w')


% remove bad ICs
% EEG = pop_subcomp(EEG,ICs2remove); 

%% 4. (Optional) Shrinkage regularization of R covariance matrix
% Increases numerical stability, particularly for reduced-rank data

nchans = EEG.nbchan;
gamma = .01;
covRr = covBroad*(1-gamma) + eye(nchans)*gamma*mean(eig(covBroad));

%--------------------------------------------------------------------------
% PLOT
%--------------------------------------------------------------------------
figure, set(gcf,'Name','Shrinkage regularization')
subplot(221)
imagesc(covTheta)
axis square
title('covTheta')
colorbar

subplot(222)
imagesc(covBroad)
axis square
title('covBroad')
set(gcf,'color','w')
colorbar
colormap hot

subplot(223)
imagesc(covRr)
axis square
title('covBroad, regularized')
set(gcf,'color','w')
colorbar
colormap hot

subplot(224)
imagesc(covRr-covBroad)
axis square
title('Regularized vs. non-regularized ')
set(gcf,'color','w')
colorbar
colormap hot

%% 5. Perform generalized eigen value decomposition

%--------------------------------------------------------------------------
% The most important line
%--------------------------------------------------------------------------
% [evecs,evalsM] = eig(covTheta,covRr);         % WITH shrinkage regularization
[evecs,evalsM] = eig(covTheta,covBroad);        % WITHOUT shrinkage regularization

% Taking the eigenvector associated with the highest eigenvalue
[evals, sidx] = sort(diag(evalsM), 'descend');
evecs = evecs(:, sidx);                   % IMPORTANT also to sort eigenvectors 
                                                % to keep the mapping between eigenvalues 
                                                % and eigenvectors intact
[~,comp2use] = max(evals);                      % Select component associated with highest eigenvalue

for v = 1:size(evecs,2)
    evecs(:,v) = evecs(:,v)/norm(evecs(:,v));   % normalize to unit length (useful when averaging between subjects)
end

figure, set(gcf,'Name','Eigenvalues')
subplot(121)
imagesc(evalsM)
axis square
title('Eigenvalues (Matlab output)')

subplot(122)
plot(evals,'-o')
axis square
title('Eigenvalues (diagonal & sorted)')
colormap(cmap)
set(gcf,'color','w')

%% 6. GED weights vs. component maps

% 6A. Plot the spatial filter weights (eigenvector)
%-----------------------------------------------------
comp2plot = 1;
figure, set(gcf,'Name','Spatial filter weights')
data2plot = evecs(:,comp2plot);
topoplot(double(squeeze(data2plot)),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')
colormap(cmap)
title(['Eigenvector ' num2str(comp2plot)])

% 6B. Compute the Spatial filter Forward model (Component maps)
%-----------------------------------------------------

% 1st OPTION (Cohen & Gulbinaite, 2016): result is channels x components
% topos = inv(evecs');                                   % works only for full-rank matrices)
topos = covTheta * evecs / (evecs' * covTheta * evecs);  % works for all kinds of matrices

% 2nd OPTION (Haufe et al., 2014): result is components x channels
% topos = evecs' * covTheta;
% comp2use = 1;
% topo  = evecs(:,comp2use)' * covTheta;                 % Single component

% 5C. Component topography sign invariance
%-----------------------------------------------------
[~,tempidx] = max(abs(topos(:,comp2use)));               % find the biggest component
topos = topos * sign(topos(tempidx,comp2use));           % force to positive sign

% Compare filter Forward model vs. Filter weights
%-----------------------------------------------------
ncomps = 2;                                              % Plotting just first 2 components
figure,set(gcf,'Name','Spatial patterns vs. spatial filters')
for compi = 1:ncomps

    subplot(2,ncomps,compi)
    
    % Plotting component maps (from 1st OPTION)
    topoplot(double(squeeze(topos(:,compi))),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')
    
    % Plotting from 2nd Option
    %     topoplot(double(squeeze(topos(compi,:))),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')
    title([num2str(compi) ' Component map'])
    
    % Plot spatial filter weights
    subplot(2,ncomps,compi+2)
    topoplot(double(squeeze(evecs(:,compi))),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')
    title([num2str(compi) ' Eigenvector'])

end
colormap(cmap)
set(gcf,'Color','w')
%% 7. Apply spatial filters to the data: pass the data through the spatial filter
ncomps = 2;
ressdata = zeros(EEG.pnts,EEG.trials,ncomps );
for compi = 1:ncomps

    for triali = 1:EEG.trials
        ressdata(:,triali,compi) = (squeeze(EEG.data(:,:,triali))'*evecs(:,compi))'; % 1st option
        % ress(:,triali,compi) = (squeeze(EEG.data(:,:,triali))'*evecs(compi,:))';   % 2nd option
    end
end
%% 8. Analyze components, e.g. compute the TF of the components

% Settings for wavelet analysis
times2save = -300:25:3200;
basetime   = [ -500 -200 ];

min_frex  =  2;
max_frex  = 40;
num_frex  = 30;
nCycRange = [4 6];

% setup convolution parameters
times2saveidx = dsearchn(EEG.times',times2save');
baseidx = dsearchn(EEG.times',basetime');

frex = logspace(log10(min_frex),log10(max_frex),num_frex);

% wavelet parameters
s = logspace(log10(nCycRange(1)),log10(nCycRange(2)),num_frex)./(2*pi.*frex);
t = -2:1/EEG.srate:2;
halfwave = floor((length(t)-1)/2);

% nData = EEG.pnts;%length(condmarks)*EEG.pnts;
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
tfRESS_all = zeros(ncomps,num_frex,length(times2save));
% tfRESS_cond = zeros(length(condlist),num_frex,length(times2save));


% Wavelet convolution 
%-----------------------------------
for compi = 1:ncomps
    disp(compi)
    data2analyze = squeeze(ressdata(:,:,compi));
    eegX = fft( reshape(data2analyze,1,nData) ,nConv);

    % loop over frequencies
    for fi=1:num_frex
        as = ifft( eegX.*cmwX(fi,:) );
        as = as(halfwave+1:end-halfwave);
        as = reshape(as,EEG.pnts,EEG.trials);

        % condition-average baseline
        basepow = mean(mean( abs(as(baseidx(1):baseidx(2),:)).^2,2),1);

        tfRESS_all(compi,fi,:) = 10*log10( mean(abs(as(times2saveidx,:)).^2,2) ./ basepow );

        % condition-specific, trial-average power
        %     for condi=1:length(condlist)
        %         tfRESS_cond(condi,fi,:) = 10*log10( mean( abs(as(times2saveidx,condlist(condi)==condmarks)).^2 ,2) / basepow );
        %     end

    end % end frequencies loop
end % end components loop
%% PLOT first GED component maps % TF results

figure, set(gcf,'Name','Component TF results')
for compi = 1:ncomps

    subplot(2,ncomps,compi)
    % Plotting from 1st Option
    topoplot(double(squeeze(topos(:,compi))),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')
    if compi ==2
        topoplot(double(squeeze(topos(:,compi))*(-1)),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')

    end
    % Plotting from 2nd Option
    %     topoplot(double(squeeze(topos(compi,:))),EEG.chanlocs, 'plotrad',.7,'electrodes','on', 'numcontour',0,'whitebk','on')
    colorbar
    title(['Comp ' num2str(compi)])

    % Plt TF results
    ax(compi+ncomps) = subplot(2,ncomps,compi+ncomps);
    contourf(times2save,frex,squeeze(tfRESS_all(compi,:,:)),100,'linecolor','none')
    set(gca,'xlim',[2000 3000],'yscale','log','ytick',round(logspace(0,log10(length(frex)),6)))
    axis square
    xlabel('Time (ms)')
    ylabel('Frequencies (Hz)')
    colorbar
end
colormap(cmap)
set(gcf,'Color','w')

%--------------------------------------------------------------------------
%%             PART 2: Flicker frequency specific filters
%--------------------------------------------------------------------------
% From the paper: "For each participant, six spatial filters were constructed
% (separately for each tagging frequency, and stimulus type) because:
% (1) SSVEP topographies differ for centrally presented target and
% peripheral flankers (Fig. 2);
% (2) different frequency SSVEPs have different sources and therefore different
% scalp projections (Heinrichs-Graham and Wilson, 2012; Lithari et al., 2016);
% (3) SSVEP topographies may differ across participants due to anatomical
% differences".

%% Explore responses to different flicker frequencies and stimuli (topographies)

% FFT settings
time4fft   = [100 2600]; % firs 500ms - cue/postcue, another 500ms - ERP; time 0 is cue onset; flicker duration 8000 ms
tidx       = dsearchn(EEG.times',time4fft');
resolution = 0.1;
nFFT       = ceil( EEG.srate/resolution ); % length of FFT based on desired freq resolution in the power spectra
hz         = EEG.srate/2*linspace(0,1,floor(nFFT/2+1));
freqs2use  = [7.5 10 15];
freqidx    = dsearchn(hz',freqs2use');
nfreqs     = length(freqs2use);

% SNR settings
skipbins = round(0.5/diff(hz(1:2)));            % skip 0.5 Hz around the peak
numbins  = round(1/diff(hz(1:2))) + skipbins;   % +-1 Hz = 10

% Initialize variable to store the FFT result
[fftpow_SNR fftpow] = deal(zeros(length(condlist),EEG.nbchan, length(hz)));

%% Perform FFT

for condi = 1:length(condlist)

    % data for this condition
    tempdata = EEG.data(:,:,condlist(condi)==condmarks);

    for chani = 1:EEG.nbchan

        temp_pow = mean(abs(fft(detrend(squeeze(tempdata(chani,tidx(1):tidx(2),:))),nFFT,1)/diff(tidx)).^2,2);
        temp_pow = temp_pow(1:length(hz));
        fftpow(condi,chani,:) = temp_pow;

        % convert to SNR
        for hzi = numbins+1:length(hz)-numbins-1
            numer = temp_pow(hzi);
            denom = mean(temp_pow([hzi-numbins:hzi-skipbins hzi+skipbins:hzi+numbins]));
            fftpow_SNR(condi,chani,hzi) = numer./denom;
        end

    end % end channel loop
end % end condition loop

%% PLOT Power
conds2use = cell(nfreqs,2);
conds2use{1,1} = [3 4];        % 7.5 Hz target
conds2use{1,2} = [1 2];        % 7.5 Hz flankers
conds2use{2,1} = [1 2 7 8];    % 10 Hz target
conds2use{2,2} = [3 4 5 6];    % 10 Hz flankers
conds2use{3,1} = [5 6];        % 15 Hz target
conds2use{3,2} = [7 8];        % 15 Hz flankers

figure, set(gcf,'Name','Power at flicker frequencies')
for freqi = 1:nfreqs
    if freqi == 1
        clim = [0 1.5];
    elseif freqi == 2
        clim = [0 2];
    elseif freqi == 3
        clim = [0 1];
    end

    subplot(2,3,freqi)
    topoplot(squeeze(mean(fftpow(conds2use{freqi,1},:,freqidx(freqi)))),EEG.chanlocs, 'plotrad',.7,'electrodes','off','maplimits',clim,'numcontour',0,'whitebk','on')
    title([num2str(freqs2use(freqi)) ' Hz target'])
    colorbar

    ax(freqi + nfreqs ) = subplot(2,nfreqs ,freqi + nfreqs );
    topoplot(squeeze(mean(fftpow(conds2use{freqi,2},:,freqidx(freqi)))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'maplimits',clim,'numcontour',0,'whitebk','on')
    title([num2str(freqs2use(freqi)) ' Hz flankers'])
    colorbar
end
colormap(cmap)

%% PLOT SNR: Target/Flankers separately
figure, set(gcf,'Name','SNR at flicker frequencies')
for freqi = 1:nfreqs
    if freqi < 3
        clim = [0 3];
    else
        clim = [0 1.6];
    end
%     clim = [0 ceil(max(squeeze(mean(fftpow_SNR(conds2use{freqi,1},:,freqidx(freqi))))))];
    subplot(2,3,freqi)
    topoplot(squeeze(mean(fftpow_SNR(conds2use{freqi,1},:,freqidx(freqi)))),EEG.chanlocs, 'plotrad',.7,'electrodes','off','maplimits',clim,'numcontour',0,'whitebk','on')
    title([num2str(freqs2use(freqi)) ' Hz target (SNR)'])
    colorbar

    ax(freqi + nfreqs ) = subplot(2,nfreqs ,freqi + nfreqs );
    topoplot(squeeze(mean(fftpow_SNR(conds2use{freqi,2},:,freqidx(freqi)))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'maplimits',clim,'numcontour',0,'whitebk','on')
    title([num2str(freqs2use(freqi)) ' Hz flankers (SNR)'])
    colorbar
end
colormap(cmap)

%%-------------------------------------------------------------------------
%%                          GED on Flicker data
%--------------------------------------------------------------------------
% NOTES:
% 1. If you have interpolated channels - need to remove it, there's no
% unique info in those.
% 2. Removing too many ICs with artifacts will reduce rank of the data
% matrix, which may compromise matrix inversions and GED solutions will not
% converge

peakwidt  = 0.5;      % FWHM at peak frequency - usually 0.5 or 0.6 works well.
neighfreq = 1;        % distance of neighboring frequencies away from peak frequency, +/- in Hz, could be increased, e.g. to 1.5 Hz
neighwidt = 1;        % FWHM of the neighboring frequencies; can also be increased; i.e. a parameter to adjust
regu      = .01;      % 1% shrinkage regularization of R covariance matrices

nFFT      = ceil( EEG.srate/.1 );
hz        = EEG.srate/2*linspace(0,1,nFFT/2+1);

% initialize matrix to store spatial filters
resstopos = zeros(2,nfreqs,EEG.nbchan); % targets/flankers x 3 freqs x channels
resspow_SNR = zeros(2,nfreqs,length(hz));

% LOOP over frequencies
for atcond = 1:2
    for freqi = 1:nfreqs

        %% Select trials based on codition (i.e. flicker frequency and type of stimulus)

        if freqi == 1 && atcond == 1                                % 7.5 Hz targets
            whichtrials = find(condmarks==20 | condmarks==21);
        elseif freqi == 2 && atcond == 1                            % 10 Hz targets
            whichtrials = find(condmarks==40 | condmarks==41 | condmarks==60 | condmarks==61 );
        elseif freqi==3 && atcond == 1                              % 15 Hz targets
            whichtrials = find(condmarks==80 | condmarks==81);
        % -----------------------------------------------------------------

        elseif freqi == 1 && atcond == 2                            % 7.5 Hz flankers
            whichtrials = find(condmarks==40 | condmarks==41);
        elseif freqi == 2 && atcond == 2                            % 10 Hz flankers
            whichtrials = find(condmarks==20 | condmarks==21 | condmarks==80 | condmarks==81 );
        elseif freqi==3 && atcond == 2                              % 15 Hz flankers
            whichtrials = find(condmarks==60 | condmarks==61);
        end

        littleN = numel(whichtrials);

        %% concatenate all data for filtering
        subdata = EEG.data(:,:,whichtrials);

        %% get covariance for flicker frequency
        nComps = EEG.nbchan;

        % filter 
        [filtTmp,empVals,fx1,hz_filt] = filterFGx(subdata,EEG.srate,freqs2use(freqi),peakwidt);

        % get flicker time periods and compute covariance
        filtTmp = reshape(filtTmp(:,tidx(1):tidx(2),:),nComps,[]);
        filtTmp = bsxfun(@minus,filtTmp,mean(filtTmp,2));
        covS = (filtTmp*filtTmp') / (diff(tidx)-1);

        %% get covariance for below flicker frequency

        % filter  
        [filtTmp,empVals,fx2,hz_filt] = filterFGx(subdata,EEG.srate,freqs2use(freqi)-neighfreq,neighwidt);

        % get flicker time periods and compute covariance
        filtTmp = reshape(filtTmp(:,tidx(1):tidx(2),:),nComps,[]);
        filtTmp = bsxfun(@minus,filtTmp,mean(filtTmp,2));
        covLo = (filtTmp*filtTmp') / (diff(tidx)-1);

        %% get covariance for above flicker frequency

        % filter 
        [filtTmp,empVals,fx3,hz_filt] = filterFGx(subdata,EEG.srate,freqs2use(freqi)+neighfreq,neighwidt);

        % get flicker time periods and compute covariance
        filtTmp = reshape(filtTmp(:,tidx(1):tidx(2),:),nComps,[]);
        filtTmp = bsxfun(@minus,filtTmp,mean(filtTmp,2));
        covHi = (filtTmp*filtTmp') / (diff(tidx)-1);

        %% Plot the filters for visual inspection
%         figure
%         plot(hz_filt,fx1,'ro-')
%         hold on
%         plot(hz_filt,fx2,'bo-')
%         plot(hz_filt,fx3,'bo-')
%         set(gca,'xlim',[max(freqs2use(freqi)-10,0) freqs2use(freqi)+10]);
%         xlabel('Frequency (Hz)'), ylabel('Amplitude gain')
%         set(gcf,'color','w')
%         box off
        %% Peform GED

        % Average two Reference covariance matrices
        covR = (covHi+covLo)/2;

        % Regularization
        gamma = .01;
        covRr = covR*(1-gamma) + eye(EEG.nbchan)*gamma*mean(eig(covR));

        % Again The most important line
        [evecs,evalsM] = eig(covS,covRr);                       % WITH shrinkage regularization

        % Taking the eigenvector associated with the highest eigenvalue
        [evals, sidx] = sort(diag(evalsM), 'descend');
        evecs = real(evecs(:, sidx));
        [~,comp2use] = max(evals);

        for v = 1:size(evecs,2)
            evecs(:,v) = evecs(:,v)/norm(evecs(:,v));           % normalize to unit length (useful when averaging between subjects)
        end

        % 1st OPTION: result is channels x components
        % topos = inv(evecs');                                   % get maps (this is fine for full-rank matrices)
        topos = covS * evecs / (evecs' * covS * evecs);          % works for all kinds of matrices

        % Component topography sign invariance
        [~,tempidx] = max(abs(topos(:,comp2use)));               % find the biggest component
        topos = topos * sign(topos(tempidx,comp2use));           % force to positive sign

        %% Algorithm to find a component with the highest SNR at the flicker frequency

        % reconstruct top 6 components: Apply spatial filter to original/unfiltered data
        tmpresstx = zeros(6,EEG.pnts,length(whichtrials));
        for triali=1:length(whichtrials)
            tmpresstx(:,:,triali) = ( subdata(:,:,triali)' * evecs(:,1:6) )';
        end

        % Compute component power spectra
        tmpresspwr = mean(abs(fft(tmpresstx(:,tidx(1):tidx(2),:),nFFT,2)).^2,3);
        tmpresspwr = tmpresspwr(:,1:length(hz));

        % Plot component topographies and power spectra
%         freqidx = dsearchn(hz',freqs2use(freqi));
%         figure
%         for compi = 1:6
% 
%             subplot(2,6,compi)
%             topoplot(double(squeeze(topos(:,compi))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'numcontour',0,'whitebk','on')
%             title(['Comp ' num2str(compi)])
% 
%             ax(compi + 6) = subplot(2,6,compi + 6);
%             plot(hz,squeeze(tmpresspwr(compi,:)),'b-','Linewidth',1.5);
%             set(gca,'xlim',[freqs2use(freqi)-7 freqs2use(freqi)+7],'ylim',[0 squeeze(tmpresspwr(compi,freqidx))*1.1])
%             xlabel('Frequencies (Hz)')
%             ylabel('Power (a.u.)')
%         end
%         colormap(cmap)
%         set(gcf,'Color','w')

        %------------------------------------------------------------------
        % Select component with highest SNR
        %------------------------------------------------------------------
        % compute SNR at the flicker frequency
        freqidx = dsearchn(hz',freqs2use(freqi));
        denom = squeeze(mean(tmpresspwr(:,[freqidx-numbins:freqidx-skipbins freqidx+skipbins:freqidx+numbins]),2));
        numer = squeeze(tmpresspwr(:,freqidx));
        tmpsnr = numer./denom;

        [~,comp2use] = max(tmpsnr);

        % Convert to SNR power spectrum from one component & store the data
        tmpresspwr = squeeze(tmpresspwr(comp2use,:));

        % convert to SNR
        for hzi = numbins+1:length(hz)-numbins-1
            numer = tmpresspwr(hzi);
            denom = mean(tmpresspwr([hzi-numbins:hzi-skipbins hzi+skipbins:hzi+numbins]));
            resspow_SNR(atcond,freqi,hzi) = numer./denom;
        end

        %% Apply filter to the data

        resstopos(atcond,freqi,:) = topos(:,comp2use);

        % apply filter to original/unfiltered data
        for triali=1:length(whichtrials)
            EEG.ress(:,whichtrials(triali)) = ( subdata(:,:,triali)' * evecs(:,comp2use) )';
        end

    end
end

%% PLOT the results

% Even thought spatial filters do not differ that much, we are optimizing
% for different frequencies and this clear from the power spectra

figure, set(gcf,'Name','Component maps across conditions')
for freqi = 1:nfreqs

    subplot(3,3,freqi)
    topoplot(double(squeeze(resstopos(1,freqi,:))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'numcontour',0,'whitebk','on')
    %             colorbar
    title([num2str(freqs2use(freqi)) ' Hz, attend'])

    ax(freqi + nfreqs) = subplot(3,3,freqi + nfreqs);
    topoplot(double(squeeze(resstopos(2,freqi,:))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'numcontour',0,'whitebk','on')
    title([num2str(freqs2use(freqi)) ' Hz, ignore'])

    ax(freqi + nfreqs*2) = subplot(3,3,freqi + nfreqs*2);
    plot(hz,squeeze(resspow_SNR(1,freqi,:)),'r-','Linewidth',1.5);
    hold on
    plot(hz,squeeze(resspow_SNR(2,freqi,:)),'b-','Linewidth',1.5);
    legend({'Attend' 'Ignore'})
    set(gca,'xlim',[5 25])
    xlabel('Frequencies (Hz)')
    ylabel('Power (SNR)')
    box off
    legend box off
end
colormap(cmap)
set(gcf,'Color','w')
%% Analyze components further, e.g. TF analysis as above

%% Overfitting 
% Compute spatial filters when from trials when the flicker frequency was
% not presented. This gives us an estimate on what to expect in terms of
% SNR due to overfitting.

freqi = 1; 

% Select appropriate trials:
if freqi == 1
    whichtrials = find(condmarks==60 | condmarks==61 | condmarks==80 | condmarks==81); % Conditions w/o 7.5 Hz in the target position
elseif freqi == 3
    whichtrials = find( condmarks==40 | condmarks==41 );   % Conditions w/o 15 Hz in the target position
elseif freqi == 2
    whichtrials = find(condmarks==20 | condmarks==21 | condmarks==80 | condmarks==81); % Conditions w/o 10 Hz in the target position

end
subdata = EEG.data(:,:,whichtrials);

%% get covariance matrices for frequency of interest (i.e. 10 Hz) and around it

% filter
[filtTmp,empVals,fx1,hz_filt] = filterFGx(subdata,EEG.srate,freqs2use(freqi),peakwidt);

% get flicker time periods and compute covariance
filtTmp = reshape(filtTmp(:,tidx(1):tidx(2),:),nComps,[]);
filtTmp = bsxfun(@minus,filtTmp,mean(filtTmp,2));
covS = (filtTmp*filtTmp') / (diff(tidx)-1);

% get covariance for R1

% filter
[filtTmp,empVals,fx2,hz_filt] = filterFGx(subdata,EEG.srate,freqs2use(freqi)-neighfreq,neighwidt);

% get flicker time periods and compute covariance
filtTmp = reshape(filtTmp(:,tidx(1):tidx(2),:),nComps,[]);
filtTmp = bsxfun(@minus,filtTmp,mean(filtTmp,2));
covLo = (filtTmp*filtTmp') / (diff(tidx)-1);

% get covariance for R2

% filter
[filtTmp,empVals,fx3,hz_filt] = filterFGx(subdata,EEG.srate,freqs2use(freqi)+neighfreq,neighwidt);

% get flicker time periods and compute covariance
filtTmp = reshape(filtTmp(:,tidx(1):tidx(2),:),nComps,[]);
filtTmp = bsxfun(@minus,filtTmp,mean(filtTmp,2));
covHi = (filtTmp*filtTmp') / (diff(tidx)-1);

%% Peform GED

% Average two Reference covariance matrices
covR = (covHi+covLo)/2;

gamma = .01;
covRr = covR*(1-gamma) + eye(EEG.nbchan)*gamma*mean(eig(covR));

% Perform GED
[evecs,evalsM] = eig(covS,covRr);       

% Taking the eigenvector associated with the highest eigenvalue
[evals, sidx] = sort(diag(evalsM), 'descend');
evecs = real(evecs(:, sidx));
[~,comp2use] = max(evals);

% Get component topography and time series
topo = covS * evecs(:,comp2use) / (evecs(:,comp2use)' * covS * evecs(:,comp2use));          % works for all kinds of matrices

ress_noise = zeros(EEG.pnts,length(whichtrials));
for triali=1:length(whichtrials)
   ress_noise(:,triali) = ( subdata(:,:,triali)' * evecs(:,comp2use) )';
end

% Perform FFT
tmpresspwr = mean(abs(fft(ress_noise(tidx(1):tidx(2),:),nFFT,1)).^2,2);
tmpresspwr = tmpresspwr(1:length(hz));
h0_snr = tmpresspwr;
% convert to SNR
for hzi = numbins+1:length(hz)-numbins-1
    numer = tmpresspwr(hzi);
    denom = mean(tmpresspwr([hzi-numbins:hzi-skipbins hzi+skipbins:hzi+numbins]));
    h0_snr(hzi) = numer./denom;
end

%% PLOT
figure, set(gcf,'Name','Overfitting')
subplot(2,3,[1:3])
plot(hz,squeeze(resspow_SNR(1,freqi,:)),'r-','Linewidth',1.5);
hold on
plot(hz,squeeze(resspow_SNR(2,freqi,:)),'b-','Linewidth',1.5);
plot(hz,h0_snr,'m-','Linewidth',1.5);
legend({[num2str(freqs2use(freqi)) ' Hz flicker attended']  [num2str(freqs2use(freqi)) ' Hz flicker ignored'] 'H0 hypothesis'})
set(gca,'xlim',[5 22],'ylim',[0 10])
xlabel('Frequencies (Hz)')
ylabel('Power (SNR)')
box off
legend box off

% Plot component maps 
subplot(234)
topoplot(double(squeeze(resstopos(1,freqi,:))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'numcontour',0,'whitebk','on')
title([num2str(freqs2use(freqi)) ' Hz flicker attended'])
colormap(cmap)

subplot(235)
topoplot(double(squeeze(resstopos(2,freqi,:))),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'numcontour',0,'whitebk','on')
title([num2str(freqs2use(freqi)) ' Hz flicker ignored'])
colormap(cmap)

subplot(236)
topoplot(double(topo),EEG.chanlocs, 'plotrad',.7,'electrodes','off', 'numcontour',0,'whitebk','on')
colormap(cmap)
title('H0 component map')
%% THE END

